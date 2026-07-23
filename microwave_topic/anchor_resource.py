"""
ResourceDiagnosisService-aligned inference —— 完全自包含、不 import 任何项目文件。

保持线上服务契约（与 resource_diagnosis_service.py 一致）：

    service = ResourceDiagnosisService(model_path, sbert_path)
    root_cause_resources, ne_to_alarm_objs = service.predict(input_json)

`input_json` 遵循资源服务 schema：alarms / resources / resourceRelations / happenRelations
`root_cause_resources` 遵循既有服务输出 schema：[{"resourceName": "...", "confidence": 0.95}, ...]

模型/特征实现内联自最终版 dirgate 推理脚本，与 infer_model.py / train_final_model.py 的
共享块逐字节一致，因此能直接加载它们训练导出的同格式 model.pt。除 SBERT 模型目录外，
仅依赖 torch / torch_geometric / sentence-transformers / numpy。
"""

import collections
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.logger import SERVICE_LOGGER
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import degree, to_dense_batch

# ==================================================================================
# ============ 内联共享实现（与 infer_model.py / train_final_model.py 逐字节一致）============
# ==================================================================================
EDGE_STAT_DIM = 12
DIR_STAT_INDICES = (2, 3, 5, 6)  # same_count, opp_count, same_ratio, opp_ratio
DEGREE_DIM = 8
LOGIT_CLIP = 30.0
ARTIFACT_FORMAT = "rca_dirgate_standalone_v1"


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # 真实数据里 link/TTid/neName 等字段会出现裸 NaN
        return ""
    return str(x).strip()


def make_undirected_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def parse_link_value(link_value: Any) -> Optional[Tuple[str, str]]:
    s = safe_str(link_value)
    if not s:
        return None
    for sep in ["<->", "-->", "->", "=>", "→", "↔", ","]:
        if sep in s:
            left, right = s.split(sep, 1)
            left, right = left.strip(), right.strip()
            return (left, right) if left and right else None
    return None


def canonicalize_vid(raw_vid: str, canonical_vid_map: Dict[str, str]) -> Optional[str]:
    if raw_vid in canonical_vid_map:
        return canonical_vid_map[raw_vid]
    return canonical_vid_map.get(raw_vid.strip())


def parse_alarm_timestamp(alarm: Dict[str, Any]) -> Optional[float]:
    time_keys = [
        "timestamp", "time", "eventTime", "occurTime", "occurrenceTime",
        "firstOccurrence", "lastOccurrence", "lastOccurTime", "createTime",
        "raiseTime", "raisedTime", "startTime", "insertTime",
    ]
    for key in time_keys:
        if key not in alarm:
            continue
        value = alarm.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (int, float)):
            ts = float(value)
            return ts / 1000.0 if ts > 1e12 else ts
        s = str(value).strip()
        if not s:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", s):
            ts = float(s)
            return ts / 1000.0 if ts > 1e12 else ts
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, OSError) as e:
            SERVICE_LOGGER.debug(f"fromisoformat parse failed for '{s}': {e}")
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"]:
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    return None


def alarm_title(alarm: Dict[str, Any]) -> str:
    for key in ["title", "name", "alarmName", "alarmTitle", "eventName", "type"]:
        v = safe_str(alarm.get(key))
        if v:
            return v
    return "[EMPTY_ALARM]"


def build_link_alarm_text(alarm: Dict[str, Any]) -> str:
    parts = [alarm_title(alarm)]
    link = safe_str(alarm.get("link"))
    src = safe_str(alarm.get("linkDirectionSource"))
    if link:
        parts.append(f"link={link}")
    if src:
        parts.append(f"linkDirectionSource={src}")
    return " | ".join(parts)


def build_edge_text(events: List[Dict[str, Any]], max_titles: int) -> str:
    if not events:
        return ""
    return " ; ".join(build_link_alarm_text(a) for a in events[:max_titles]) or "[EMPTY_LINK_ALARM]"


def compute_edge_stats(events: List[Dict[str, Any]], src_vid: str, dst_vid: str,
                       ref_ts: Optional[float], cmap: Dict[str, str]) -> np.ndarray:
    stat = np.zeros(EDGE_STAT_DIM, dtype=np.float32)
    total = len(events)
    if total == 0:
        return stat
    sys_cnt = same_cnt = opp_cnt = unk_cnt = 0
    timestamps: List[float] = []
    for alarm in events:
        if safe_str(alarm.get("linkDirectionSource")).upper() == "SYS_LINK":
            sys_cnt += 1
        parsed = parse_link_value(alarm.get("link"))
        if parsed is None:
            unk_cnt += 1
        else:
            a = canonicalize_vid(parsed[0], cmap)
            b = canonicalize_vid(parsed[1], cmap)
            if a is not None and b is not None and a == src_vid and b == dst_vid:
                same_cnt += 1
            elif a is not None and b is not None and a == dst_vid and b == src_vid:
                opp_cnt += 1
            else:
                unk_cnt += 1
        ts = parse_alarm_timestamp(alarm)
        if ts is not None:
            timestamps.append(ts)
    denom = max(total, 1)
    stat[0] = np.log1p(total)
    stat[1] = np.log1p(sys_cnt)
    stat[2] = np.log1p(same_cnt)
    stat[3] = np.log1p(opp_cnt)
    stat[4] = np.log1p(unk_cnt)
    stat[5] = same_cnt / denom
    stat[6] = opp_cnt / denom
    stat[7] = unk_cnt / denom
    stat[8] = 1.0
    if timestamps and ref_ts is not None:
        rec = [max(0.0, (ref_ts - ts) / 60.0) for ts in timestamps]
        stat[9] = np.log1p(min(rec)) / 10.0
        stat[10] = np.log1p(float(np.mean(rec))) / 10.0
        stat[11] = np.log1p(max(0.0, (max(timestamps) - min(timestamps)) / 60.0)) / 10.0
    return np.nan_to_num(stat, nan=0.0, posinf=1e4, neginf=-1e4)


class TextEncoder:
    """SentenceTransformer 包装。"""

    def __init__(self, model_path: str, device: str, batch_size: int = 128, normalize: bool = False) -> None:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Loading SBERT: {model_path} on {device}")
        self.model = SentenceTransformer(model_path, device=device)
        dim = (self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension")
               else self.model.get_sentence_embedding_dimension())
        self.dim = int(dim or 384)
        self.batch_size = batch_size
        self.normalize = normalize
        print(f"[INFO] SBERT dim = {self.dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                show_progress_bar=False, normalize_embeddings=self.normalize)
        return np.nan_to_num(np.asarray(emb, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)

    def close(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DirGatedTransformerConv(TransformerConv):
    """TransformerConv + same/opp 方向乘性门控（聚合前给每条边消息乘 gate∈(0,2)，无方向证据的边 gate=1）。"""

    def __init__(self, in_channels: int, out_channels: int, gate_in_dim: int,
                 gate_hidden: int = 16, **kwargs: Any) -> None:
        super().__init__(in_channels, out_channels, **kwargs)
        rng = torch.get_rng_state()
        self.gate_mlp = nn.Sequential(nn.Linear(gate_in_dim, gate_hidden), nn.ReLU(),
                                      nn.Linear(gate_hidden, self.heads))
        torch.set_rng_state(rng)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self._edge_gate: Optional[torch.Tensor] = None

    def forward(self, x, edge_index, edge_attr=None, dir_feat=None):  # type: ignore[override]
        if dir_feat is None:
            self._edge_gate = None
        else:
            raw = 2.0 * torch.sigmoid(self.gate_mlp(dir_feat))
            has_dir = (dir_feat.abs().sum(dim=-1, keepdim=True) > 0).to(raw.dtype)
            self._edge_gate = has_dir * raw + (1.0 - has_dir)
        return super().forward(x, edge_index, edge_attr)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):  # type: ignore[override]
        if self._edge_gate is not None:
            inputs = inputs * self._edge_gate.view(-1, self.heads, 1)
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)


class DiagnosisModel(nn.Module):
    """edge-aware + dirgate：degree 特征 + 双通道 edge encoder + dirgate GNN + 图内 Transformer + 分类头。"""

    def __init__(self, in_dim: int, edge_text_dim: int, edge_stat_dim: int = EDGE_STAT_DIM,
                 hidden_dim: int = 128, gnn_layers: int = 3, gnn_heads: int = 2,
                 transformer_layers: int = 2, nhead: int = 4, dropout: float = 0.1,
                 max_nodes: int = 200) -> None:
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.edge_text_dim = int(edge_text_dim)
        self.edge_stat_dim = int(edge_stat_dim)
        self.dropout = float(dropout)
        self.degree_encoder = nn.Sequential(nn.Linear(1, DEGREE_DIM), nn.ReLU(), nn.Linear(DEGREE_DIM, DEGREE_DIM))
        model_in = int(in_dim) + DEGREE_DIM
        t = hidden_dim // 2
        s = hidden_dim - t
        self.edge_text_proj = nn.Sequential(nn.LayerNorm(self.edge_text_dim), nn.Linear(self.edge_text_dim, t),
                                            nn.ReLU(), nn.Dropout(dropout))
        self.edge_stat_proj = nn.Sequential(nn.LayerNorm(self.edge_stat_dim), nn.Linear(self.edge_stat_dim, s),
                                            nn.ReLU(), nn.Dropout(dropout))
        self.edge_fuse = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.no_info_edge = nn.Parameter(torch.zeros(hidden_dim))
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(DirGatedTransformerConv(model_in, hidden_dim, gate_in_dim=len(DIR_STAT_INDICES),
                                                       heads=gnn_heads, concat=False, edge_dim=hidden_dim))
        for _ in range(gnn_layers - 1):
            self.gnn_layers.append(DirGatedTransformerConv(hidden_dim, hidden_dim, gate_in_dim=len(DIR_STAT_INDICES),
                                                           heads=gnn_heads, concat=False, edge_dim=hidden_dim))
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(gnn_layers)])
        if transformer_layers > 0:
            enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True,
                                             dim_feedforward=hidden_dim * 4, dropout=dropout,
                                             activation="gelu", norm_first=True)
            self.transformer: Optional[nn.Module] = nn.TransformerEncoder(enc, transformer_layers)
        else:
            self.transformer = None
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        _, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype).view(-1, 1)
        x = torch.cat([x, self.degree_encoder(torch.log(deg + 1.0))], dim=-1)
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_text_dim + self.edge_stat_dim))
        else:
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)
        text = edge_attr[:, :self.edge_text_dim]
        stat = edge_attr[:, self.edge_text_dim:self.edge_text_dim + self.edge_stat_dim]
        fused = self.edge_fuse(torch.cat([self.edge_text_proj(text), self.edge_stat_proj(stat)], dim=-1))
        has_alarm = (edge_attr.abs().sum(dim=-1, keepdim=True) > 0).to(fused.dtype)
        edge_emb = has_alarm * fused + (1.0 - has_alarm) * self.no_info_edge
        dir_feat = edge_attr[:, [self.edge_text_dim + i for i in DIR_STAT_INDICES]]
        for li, layer in enumerate(self.gnn_layers):
            x_in = x
            h = layer(x, edge_index, edge_emb, dir_feat)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if x_in.size(-1) == h.size(-1):
                h = h + x_in
            h = self.gnn_norms[li](h)
            x = h
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
        x_trans = self.transformer(x_dense, src_key_padding_mask=~mask) if self.transformer is not None else x_dense
        logits = self.classifier(x_trans).squeeze(-1)[mask]
        return logits.clamp(min=-LOGIT_CLIP, max=LOGIT_CLIP)


def load_model_artifact(model_path: Any, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    try:
        payload = torch.load(Path(model_path), map_location=str(device), weights_only=False)
    except TypeError:
        payload = torch.load(Path(model_path), map_location=str(device))
    arch = payload["arch"]
    model = DiagnosisModel(
        in_dim=arch["in_dim"], edge_text_dim=arch["edge_text_dim"],
        edge_stat_dim=arch.get("edge_stat_dim", EDGE_STAT_DIM),
        hidden_dim=arch["hidden_dim"], gnn_layers=arch["gnn_layers"], gnn_heads=arch["gnn_heads"],
        transformer_layers=arch["transformer_layers"], nhead=arch["nhead"], dropout=arch["dropout"],
        max_nodes=arch["max_nodes"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def select_device(arg: str) -> torch.device:
    arg = str(arg).lower()
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退 CPU。")
        return torch.device("cpu")
    return torch.device(arg)


def resolve_sbert_device(arg: str, train_device: torch.device) -> str:
    arg = str(arg).lower()
    if arg == "auto":
        return str(train_device) if train_device.type == "cuda" else "cpu"
    if arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return arg


class ResourceDiagnosisService:
    """Service-compatible inference class using the finalized edge-aware model."""

    def __init__(
            self,
            model_path: str,
            sbert_path: str = "",
            *,
            device: str = "cpu",
            sbert_device: str = "cpu",
            threshold: float = -1.0,
            top_k: int = 0,
            sbert_batch_size: int = 128,
    ) -> None:
        self.device = select_device(device)
        self.model, self.payload = load_model_artifact(model_path, device=self.device)
        self.model.eval()

        build_cfg = self.payload.get("build", {}) if isinstance(self.payload.get("build", {}), dict) else {}
        model_sbert_path = str(self.payload.get("sbert_path") or "")
        resolved_sbert_path = sbert_path or model_sbert_path
        if not resolved_sbert_path:
            raise ValueError("sbert_path is required when model artifact does not record one.")

        self.threshold = float(threshold if threshold >= 0 else self.payload.get("eval_threshold", 0.5))
        self.top_k = int(top_k or 0)
        self.max_edge_alarm_titles = int(build_cfg.get("max_edge_alarm_titles", 32))

        self.sbert = TextEncoder(
            resolved_sbert_path,
            device=resolve_sbert_device(sbert_device, self.device),
            batch_size=int(sbert_batch_size),
            normalize=bool(build_cfg.get("normalize_embeddings", False)),
        )
        expected_dim = int(
            self.payload.get("sbert_dim")
            or self.payload.get("arch", {}).get("edge_text_dim")
            or self.sbert.dim
        )
        if int(self.sbert.dim) != expected_dim:
            self.sbert.close()
            raise ValueError(f"SBERT dim mismatch: current={self.sbert.dim}, trained={expected_dim}.")

    def close(self) -> None:
        self.sbert.close()

    def predict(self, input_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Return the same shape as resource_diagnosis_service.py:
            ([{"resourceName": str, "confidence": float}], ne_to_alarm_objs)
        """
        graph_data, ne_names, ne_to_alarm_objs = self._build_graph(input_json)
        if graph_data is None:
            return [], dict(ne_to_alarm_objs)

        probs = self._get_probabilities(graph_data)
        target_indices = self._select_target_indices(probs, ne_names)
        root_cause_resources = [
            {"resourceName": ne_names[i], "confidence": round(float(probs[i]), 4)}
            for i in target_indices
        ]
        return root_cause_resources, dict(ne_to_alarm_objs)

    def _build_graph(
            self,
            input_json: Dict[str, Any],
    ) -> Tuple[Optional[Data], List[str], DefaultDict[str, List[Dict[str, Any]]]]:
        # 1. 提取基础数据
        alarms = [a for a in input_json.get("alarms", []) or [] if isinstance(a, dict)]
        resources = [r for r in input_json.get("resources", []) or [] if isinstance(r, dict)]
        res_rels = [r for r in input_json.get("resourceRelations", []) or [] if isinstance(r, dict)]
        happen_rels = [h for h in input_json.get("happenRelations", []) or [] if isinstance(h, dict)]

        # 2. 提取映射关系
        vid_to_ne = self._extract_vid_to_ne(resources)
        alarm_vid_to_ne = self._extract_alarm_vid_to_ne(happen_rels, vid_to_ne)
        ne_alarm_texts, ne_to_alarm_objs = self._collect_alarm_texts(alarms, alarm_vid_to_ne)

        # 3. 选择入图网元
        ne_names = self._select_valid_ne_names(resources, ne_alarm_texts)
        if not ne_names:
            return None, [], ne_to_alarm_objs

        # 4. 构建网元索引和物理边
        ne_to_idx = {name: idx for idx, name in enumerate(ne_names)}
        canonical_ne = {name.strip(): name for name in ne_names}
        physical_edges, physical_edge_keys = self._collect_physical_edges(res_rels, vid_to_ne, ne_to_idx)

        # 5. 遍历告警，收集节点文本和边事件
        node_alarm_texts, edge_events, ref_ts = self._collect_node_texts_and_edge_events(
            alarms, alarm_vid_to_ne, ne_to_idx, canonical_ne, physical_edge_keys, ne_names)

        # 6. 构建节点特征
        x = self._build_node_features(node_alarm_texts)

        # 7. 处理无物理边的情况
        if not physical_edges:
            physical_edges = [(0, 0)]

        # 8. 构建边文本嵌入
        edge_text_emb = self._build_edge_text_embeddings(physical_edges, edge_events)

        # 9. 构建边索引和属性
        edge_index, edge_attr = self._build_edge_index_and_attr(
            physical_edges, edge_events, edge_text_emb, ref_ts, ne_names, canonical_ne)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), ne_names, ne_to_alarm_objs

    def _collect_node_texts_and_edge_events(
            self,
            alarms: List[Dict[str, Any]],
            alarm_vid_to_ne: Dict[str, str],
            ne_to_idx: Dict[str, int],
            canonical_ne: Dict[str, str],
            physical_edge_keys: set,
            ne_names: List[str],
    ) -> Tuple[List[List[str]], Dict[Tuple[int, int], List[Dict[str, Any]]], Optional[float]]:
        """遍历告警，收集节点告警文本、边事件和参考时间戳"""
        node_alarm_texts: List[List[str]] = [[] for _ in ne_names]
        edge_events: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        timestamps: List[float] = []

        for alarm in alarms:
            ne_name = alarm_vid_to_ne.get(safe_str(alarm.get("alarmVertexVid"))) or safe_str(alarm.get("neName"))
            ne_idx = ne_to_idx.get(ne_name)
            if ne_idx is None:
                continue

            ts = parse_alarm_timestamp(alarm)
            if ts is not None:
                timestamps.append(ts)

            alarm = self._normalize_link(alarm, ne_name)
            edge_key = self._edge_key_from_link_alarm(alarm, canonical_ne, ne_to_idx, physical_edge_keys)
            if edge_key is not None:
                edge_events.setdefault(edge_key, []).append(alarm)
            else:
                node_alarm_texts[ne_idx].append(alarm_title(alarm))

        ref_ts = max(timestamps) if timestamps else None
        return node_alarm_texts, edge_events, ref_ts

    def _build_node_features(self, node_alarm_texts: List[List[str]]) -> torch.Tensor:
        """构建节点特征：告警文本编码"""
        node_texts = [" ".join(texts).strip() or "No Alarm" for texts in node_alarm_texts]
        return torch.tensor(self.sbert.encode(node_texts), dtype=torch.float)

    def _build_edge_text_embeddings(
            self,
            physical_edges: List[Tuple[int, int]],
            edge_events: Dict[Tuple[int, int], List[Dict[str, Any]]],
    ) -> np.ndarray:
        """为每条物理边构建文本嵌入"""
        edge_text_emb = np.zeros((len(physical_edges), self.sbert.dim), dtype=np.float32)
        texts_to_encode: List[str] = []
        edge_positions: List[int] = []

        for pos, edge in enumerate(physical_edges):
            events = edge_events.get(make_undirected_key(*edge), [])
            if events:
                texts_to_encode.append(build_edge_text(events, self.max_edge_alarm_titles))
                edge_positions.append(pos)

        if texts_to_encode:
            for pos, emb in zip(edge_positions, self.sbert.encode(texts_to_encode)):
                edge_text_emb[pos] = emb

        return edge_text_emb

    def _build_edge_index_and_attr(
            self,
            physical_edges: List[Tuple[int, int]],
            edge_events: Dict[Tuple[int, int], List[Dict[str, Any]]],
            edge_text_emb: np.ndarray,
            ref_ts: Optional[float],
            ne_names: List[str],
            canonical_ne: Dict[str, str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建边索引和边属性张量"""
        srcs: List[int] = []
        dsts: List[int] = []
        edge_attrs: List[np.ndarray] = []

        for pos, (u, v) in enumerate(physical_edges):
            events = edge_events.get(make_undirected_key(u, v), [])
            for src, dst in ((u, v), (v, u)):
                srcs.append(src)
                dsts.append(dst)
                stats = compute_edge_stats(events, ne_names[src], ne_names[dst], ref_ts, canonical_ne)
                edge_attrs.append(np.concatenate([edge_text_emb[pos], stats], axis=0))

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_attr = torch.tensor(
            np.nan_to_num(np.stack(edge_attrs, axis=0), nan=0.0, posinf=1e4, neginf=-1e4),
            dtype=torch.float,
        )
        return edge_index, edge_attr

    @staticmethod
    def _extract_vid_to_ne(resources: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        return {
            safe_str(r.get("resourceVid")): safe_str(r.get("neName"))
            for r in resources
            if safe_str(r.get("resourceVid")) and safe_str(r.get("neName"))
        }

    @staticmethod
    def _extract_alarm_vid_to_ne(happen_rels: Sequence[Dict[str, Any]], vid_to_ne: Dict[str, str]) -> Dict[str, str]:
        return {
            safe_str(h.get("srcVid")): vid_to_ne[safe_str(h.get("dstVid"))]
            for h in happen_rels
            if safe_str(h.get("srcVid")) and vid_to_ne.get(safe_str(h.get("dstVid")))
        }

    @staticmethod
    def _collect_alarm_texts(
            alarms: Sequence[Dict[str, Any]],
            alarm_vid_to_ne: Dict[str, str],
    ) -> Tuple[DefaultDict[str, List[str]], DefaultDict[str, List[Dict[str, Any]]]]:
        ne_alarm_texts: DefaultDict[str, List[str]] = collections.defaultdict(list)
        ne_to_alarm_objs: DefaultDict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for alarm in alarms:
            ne_name = alarm_vid_to_ne.get(safe_str(alarm.get("alarmVertexVid"))) or safe_str(alarm.get("neName"))
            if ne_name:
                ne_alarm_texts[ne_name].append(alarm_title(alarm))
                ne_to_alarm_objs[ne_name].append(alarm)
        return ne_alarm_texts, ne_to_alarm_objs

    @staticmethod
    def _select_valid_ne_names(resources: Sequence[Dict[str, Any]], ne_alarm_texts: Dict[str, List[str]]) -> List[str]:
        """入图节点 = 有告警网元 ∪ 无告警但 domain==Data 的网元（沿用线上服务规则）。"""
        alarm_ne_names = set(ne_alarm_texts.keys())
        data_no_alarm_ne_names = {
            safe_str(r.get("neName"))
            for r in resources
            if safe_str(r.get("neName"))
               and safe_str(r.get("neName")) not in alarm_ne_names
               and safe_str(r.get("domain")).lower() == "data"
        }
        selected = sorted(alarm_ne_names | data_no_alarm_ne_names)
        SERVICE_LOGGER.info(
            "[select] resources=%s alarm_nodes=%s data_no_alarm_nodes=%s selected=%s",
            len(resources),
            len(alarm_ne_names),
            len(data_no_alarm_ne_names),
            selected,
        )
        return selected

    @staticmethod
    def _collect_physical_edges(
            res_rels: Sequence[Dict[str, Any]],
            vid_to_ne: Dict[str, str],
            ne_to_idx: Dict[str, int],
    ) -> Tuple[List[Tuple[int, int]], set[Tuple[int, int]]]:
        edges: List[Tuple[int, int]] = []
        keys: set[Tuple[int, int]] = set()
        for rel in res_rels:
            u_ne = vid_to_ne.get(safe_str(rel.get("srcVid")))
            v_ne = vid_to_ne.get(safe_str(rel.get("dstVid")))
            if u_ne not in ne_to_idx or v_ne not in ne_to_idx or u_ne == v_ne:
                continue
            u, v = ne_to_idx[u_ne], ne_to_idx[v_ne]
            key = make_undirected_key(u, v)
            if key not in keys:
                keys.add(key)
                edges.append((u, v))
        return edges, keys

    @staticmethod
    def _normalize_link(alarm: Dict[str, Any], src_ne: str) -> Dict[str, Any]:
        """统一 link 方向来源，使新旧两种数据格式都能复用下游方向/文本逻辑：
        - 旧格式：alarm 自带 link="A->B"，原样返回；
        - 新格式：只给 dstNe（目标网元），按"源网元(neName)->dstNe"合成 link。
        返回浅拷贝，不修改调用方传入的原始 alarm（保证 ne_to_alarm_objs 仍为干净对象）。
        """
        if safe_str(alarm.get("link")):
            return alarm
        dst = safe_str(alarm.get("dstNe"))
        if not dst or not src_ne or dst == src_ne:
            return alarm  # 无 dstNe / 自环 -> 不合成，退化为节点告警
        merged = dict(alarm)
        merged["link"] = f"{src_ne}->{dst}"
        return merged

    @staticmethod
    def _edge_key_from_link_alarm(
            alarm: Dict[str, Any],
            canonical_ne: Dict[str, str],
            ne_to_idx: Dict[str, int],
            physical_edge_keys: set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        parsed = parse_link_value(alarm.get("link"))
        if parsed is None:
            return None
        left = canonicalize_vid(parsed[0], canonical_ne)
        right = canonicalize_vid(parsed[1], canonical_ne)
        if left not in ne_to_idx or right not in ne_to_idx:
            return None
        key = make_undirected_key(ne_to_idx[left], ne_to_idx[right])
        return key if key in physical_edge_keys else None

    def _get_probabilities(self, graph_data: Data) -> np.ndarray:
        graph_data = graph_data.to(self.device)
        if int(graph_data.num_nodes) > int(self.model.max_nodes):
            SERVICE_LOGGER.warning(
                "Input graph nodes=%s exceed model.max_nodes=%s; expanding padding width for inference.",
                int(graph_data.num_nodes),
                int(self.model.max_nodes),
            )
            self.model.max_nodes = int(graph_data.num_nodes)
        batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(graph_data.x, graph_data.edge_index, batch, getattr(graph_data, "edge_attr", None))
            probs = torch.sigmoid(torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9))
        return probs.detach().cpu().numpy()

    def _select_target_indices(self, probs: np.ndarray, ne_names: Sequence[str]) -> List[int]:
        if probs.size == 0:
            return []
        if self.top_k > 0:
            k = min(self.top_k, len(probs))
            return [int(i) for i in np.argsort(-probs)[:k]]

        selected = [int(i) for i, prob in enumerate(probs) if float(prob) >= self.threshold]
        if selected:
            return selected

        max_idx = int(np.argmax(probs))
        SERVICE_LOGGER.info("Fallback: selecting %s (prob=%.4f)", ne_names[max_idx], float(probs[max_idx]))
        return [max_idx]
