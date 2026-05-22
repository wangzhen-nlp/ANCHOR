from dataclasses import asdict, dataclass
from math import sqrt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    F = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from fault_grouping.alarm_events.io import is_clear_alarm
from alarm_flow_isahp.ne_topology import PAIR_FEATURE_NAMES
from alarm_flow_isahp.sequences import alarm_type_from_title


def require_torch():
    if torch is None:
        raise RuntimeError(
            "AlarmFlow ISAHP requires PyTorch. Run this command in a Python "
            "environment that has torch installed."
        ) from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class AlarmISAHPConfig:
    n_types: int
    n_alarm_sources: int
    n_alarm_types: int
    alarm_source_embedding_dim: int = 27
    alarm_type_embedding_dim: int = 4
    topology_pair_feature_dim: int = 0
    history_window_sec: float = 900.0
    time_scale_sec: float = 60.0
    hidden_size: int = 32
    num_heads: int = 4
    dropout: float = 0.0
    eps: float = 1e-8

    def __post_init__(self):
        if self.n_types < 1:
            raise ValueError("n_types must be positive")
        if self.n_alarm_sources < 1 or self.n_alarm_types < 1:
            raise ValueError("n_alarm_sources and n_alarm_types must be positive")
        if self.alarm_source_embedding_dim < 1 or self.alarm_type_embedding_dim < 1:
            raise ValueError("alarm embedding dims must be positive")
        if self.topology_pair_feature_dim < 0:
            raise ValueError("topology_pair_feature_dim must be >= 0")
        if self.history_window_sec <= 0:
            raise ValueError("history_window_sec must be positive")
        if self.time_scale_sec <= 0:
            raise ValueError("time_scale_sec must be positive")
        feature_size = 1 + self.alarm_source_embedding_dim + self.alarm_type_embedding_dim
        if self.hidden_size != feature_size:
            raise ValueError(
                "hidden_size must equal 1 + alarm_source_embedding_dim + "
                "alarm_type_embedding_dim"
            )
        if self.num_heads < 2 or self.num_heads % 2:
            raise ValueError("num_heads must be a positive even number")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**dict(payload))


class AlarmTargetWindowDataset(Dataset):
    def __init__(self, sequences):
        require_torch()
        self.windows = [
            window
            for sequence in sequences
            for window in sequence.target_windows
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def collate_alarm_target_windows(windows):
    require_torch()
    if not windows:
        raise ValueError("cannot collate an empty alarm target window batch")

    history_lengths = torch.tensor([len(window) for window in windows], dtype=torch.long)
    max_history_length = max(1, int(history_lengths.max().item()))
    history_mask = torch.zeros(len(windows), max_history_length, dtype=torch.bool)
    history_times = torch.zeros(len(windows), max_history_length, dtype=torch.float32)
    history_dts = torch.zeros(len(windows), max_history_length, dtype=torch.float32)
    history_type_ids = torch.zeros(len(windows), max_history_length, dtype=torch.long)
    history_alarm_source_ids = torch.zeros(len(windows), max_history_length, dtype=torch.long)
    history_alarm_type_ids = torch.zeros(len(windows), max_history_length, dtype=torch.long)
    topology_pair_features = None
    topology_feature_dim = max(
        (
            len(pair_features)
            for window in windows
            for pair_features in window.topology_pair_features
        ),
        default=0,
    )
    if topology_feature_dim:
        topology_pair_features = torch.zeros(
            len(windows),
            max_history_length,
            topology_feature_dim,
            dtype=torch.float32,
        )
    for row, window in enumerate(windows):
        length = len(window)
        if not length:
            continue
        history_mask[row, :length] = True
        history_times[row, :length] = torch.tensor(window.history_times, dtype=torch.float32)
        history_dts[row, :length] = torch.tensor(window.history_dts, dtype=torch.float32)
        history_type_ids[row, :length] = torch.tensor(window.history_type_ids, dtype=torch.long)
        history_alarm_source_ids[row, :length] = torch.tensor(
            window.history_alarm_source_ids,
            dtype=torch.long,
        )
        history_alarm_type_ids[row, :length] = torch.tensor(
            window.history_alarm_type_ids,
            dtype=torch.long,
        )
        if topology_pair_features is not None and window.topology_pair_features:
            topology_pair_features[row, :length] = torch.tensor(
                window.topology_pair_features,
                dtype=torch.float32,
            )
    return {
        "target_type_ids": torch.tensor(
            [window.target_type_id for window in windows],
            dtype=torch.long,
        ),
        "target_times": torch.tensor([window.target_time for window in windows], dtype=torch.float32),
        "interval_dts": torch.tensor([window.interval_dt for window in windows], dtype=torch.float32),
        "query_dts": torch.tensor([window.query_dt for window in windows], dtype=torch.float32),
        "query_alarm_source_ids": torch.tensor(
            [window.query_alarm_source_id for window in windows],
            dtype=torch.long,
        ),
        "query_alarm_type_ids": torch.tensor(
            [window.query_alarm_type_id for window in windows],
            dtype=torch.long,
        ),
        "history_times": history_times,
        "history_dts": history_dts,
        "history_type_ids": history_type_ids,
        "history_alarm_source_ids": history_alarm_source_ids,
        "history_alarm_type_ids": history_alarm_type_ids,
        "history_mask": history_mask,
        "history_lengths": history_lengths,
        "topology_pair_features": topology_pair_features,
        "windows": list(windows),
    }


class _InstanceAttention(nn.Module if nn else object):
    def __init__(self, config):
        require_torch()
        super().__init__()
        self.config = config
        self.d_head = config.hidden_size // config.num_heads
        self.half_heads = config.num_heads // 2
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.mu_proj = nn.Linear(config.hidden_size, config.n_types)
        pair_feature_dim = self.half_heads * self.d_head + config.topology_pair_feature_dim
        self.alpha_proj = nn.Linear(pair_feature_dim, config.n_types)
        self.gamma_proj = nn.Linear(pair_feature_dim, config.n_types)
        self.dropout = nn.Dropout(config.dropout)

    def _split_history_heads(self, tensor):
        batch_size, length, _hidden = tensor.shape
        return tensor.reshape(batch_size, length, self.config.num_heads, self.d_head).transpose(1, 2)

    def _split_query_heads(self, tensor):
        batch_size, _hidden = tensor.shape
        return tensor.reshape(batch_size, self.config.num_heads, self.d_head)

    def forward(self, query_features, history_features, history_mask, topology_pair_features=None):
        q = self._split_query_heads(self.q_proj(query_features))
        k = self._split_history_heads(self.k_proj(history_features))
        v = self._split_history_heads(self.v_proj(history_features))
        scores = torch.einsum("bhd,bhld->bhl", q, k) / sqrt(self.d_head)
        scores = scores.masked_fill(~history_mask[:, None, :], -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        attn = attn.masked_fill(~history_mask[:, None, :], 0.0)

        context = torch.einsum("bhl,bhld->bhd", attn, v).reshape(
            query_features.shape[0],
            self.config.hidden_size,
        )
        dynamic_mu = torch.sigmoid(self.mu_proj(context))

        pair_values = attn[..., None] * v
        alpha_pairs = pair_values[:, : self.half_heads]
        gamma_pairs = pair_values[:, self.half_heads :]
        batch_size, _heads, history_len, _d_head = alpha_pairs.shape
        alpha_pairs = (
            alpha_pairs.permute(0, 2, 1, 3)
            .reshape(batch_size, history_len, self.half_heads * self.d_head)
        )
        gamma_pairs = (
            gamma_pairs.permute(0, 2, 1, 3)
            .reshape(batch_size, history_len, self.half_heads * self.d_head)
        )
        if self.config.topology_pair_feature_dim:
            if topology_pair_features is None:
                topology_pair_features = alpha_pairs.new_zeros(
                    batch_size,
                    history_len,
                    self.config.topology_pair_feature_dim,
                )
            alpha_pairs = torch.cat([alpha_pairs, topology_pair_features], dim=-1)
            gamma_pairs = torch.cat([gamma_pairs, topology_pair_features], dim=-1)
        alpha = F.softplus(self.alpha_proj(alpha_pairs))
        gamma = F.softplus(self.gamma_proj(gamma_pairs)) + self.config.eps
        pair_mask = history_mask[..., None]
        return dynamic_mu, alpha.masked_fill(~pair_mask, 0.0), gamma.masked_fill(~pair_mask, 0.0)


class AlarmFlowISAHP(nn.Module if nn else object):
    """ISAHP-style event model for bounded alarm target windows.

    One batch row predicts one observed target alarm from the strictly earlier
    alarms kept in that target's bounded history window.
    """

    def __init__(self, config):
        require_torch()
        super().__init__()
        self.config = config
        self.alarm_source_embedding = nn.Embedding(
            config.n_alarm_sources,
            config.alarm_source_embedding_dim,
        )
        self.alarm_type_embedding = nn.Embedding(
            config.n_alarm_types,
            config.alarm_type_embedding_dim,
        )
        self.attention = _InstanceAttention(config)
        self.baseline_logits = nn.Parameter(torch.zeros(config.n_types))

    def _event_features(self, dts, alarm_source_ids, alarm_type_ids):
        dt_features = dts.clamp_min(0.0)[..., None]
        alarm_source_features = self.alarm_source_embedding(alarm_source_ids)
        alarm_type_features = self.alarm_type_embedding(alarm_type_ids)
        return torch.cat([dt_features, alarm_source_features, alarm_type_features], dim=-1)

    def forward(
        self,
        query_dts,
        query_alarm_source_ids,
        query_alarm_type_ids,
        history_dts,
        history_alarm_source_ids,
        history_alarm_type_ids,
        history_mask,
        topology_pair_features=None,
    ):
        if history_dts.shape[1] < 1:
            raise ValueError("ISAHP target windows must keep one padded history slot")
        dynamic_mu, alpha, gamma = self.attention(
            self._event_features(query_dts, query_alarm_source_ids, query_alarm_type_ids),
            self._event_features(
                history_dts,
                history_alarm_source_ids,
                history_alarm_type_ids,
            ),
            history_mask,
            topology_pair_features=topology_pair_features,
        )
        baseline = F.softplus(self.baseline_logits)[None, :] + self.config.eps
        return baseline + dynamic_mu, alpha, gamma, history_mask

    def intensity_at_events(
        self,
        target_times,
        query_dts,
        query_alarm_source_ids,
        query_alarm_type_ids,
        history_times,
        history_dts,
        history_alarm_source_ids,
        history_alarm_type_ids,
        history_mask,
        topology_pair_features=None,
    ):
        mu, alpha, gamma, pair_mask = self.forward(
            query_dts,
            query_alarm_source_ids,
            query_alarm_type_ids,
            history_dts,
            history_alarm_source_ids,
            history_alarm_type_ids,
            history_mask,
            topology_pair_features=topology_pair_features,
        )
        history_dt = (target_times[:, None] - history_times).clamp_min(0.0)
        decayed = alpha * gamma * torch.exp(-gamma * history_dt[..., None])
        decayed = decayed.masked_fill(~pair_mask[..., None], 0.0)
        intensities = mu + decayed.sum(dim=1)
        return intensities, mu, alpha, gamma, pair_mask

    def negative_log_likelihood(
        self,
        target_type_ids,
        target_times,
        interval_dts,
        query_dts,
        query_alarm_source_ids,
        query_alarm_type_ids,
        history_times,
        history_dts,
        history_alarm_source_ids,
        history_alarm_type_ids,
        history_mask,
        topology_pair_features=None,
        *,
        num_mc_samples=20,
    ):
        intensities, mu, alpha, gamma, pair_mask = self.intensity_at_events(
            target_times,
            query_dts,
            query_alarm_source_ids,
            query_alarm_type_ids,
            history_times,
            history_dts,
            history_alarm_source_ids,
            history_alarm_type_ids,
            history_mask,
            topology_pair_features=topology_pair_features,
        )
        event_intensities = intensities.gather(-1, target_type_ids[:, None]).squeeze(-1)
        log_term = torch.log(event_intensities.clamp_min(self.config.eps)).sum()

        interval_dts = interval_dts.clamp_min(0.0)
        sample_offsets = torch.rand(
            target_times.shape[0],
            num_mc_samples,
            dtype=target_times.dtype,
            device=target_times.device,
        )
        sample_times = target_times[:, None] - interval_dts[:, None] + sample_offsets * interval_dts[:, None]
        history_dt = (sample_times[:, :, None] - history_times[:, None, :]).clamp_min(0.0)
        sampled_decayed = (
            alpha[:, None, :, :]
            * gamma[:, None, :, :]
            * torch.exp(-gamma[:, None, :, :] * history_dt[..., None])
        )
        sampled_decayed = sampled_decayed.masked_fill(~pair_mask[:, None, :, None], 0.0)
        sampled_intensities = mu[:, None, :] + sampled_decayed.sum(dim=2)
        total_intensity = sampled_intensities.sum(dim=-1).mean(dim=-1)
        integral = (interval_dts * total_intensity).sum()

        denominator = target_type_ids.new_tensor(target_type_ids.numel()).clamp_min(1)
        nll = (integral - log_term) / denominator
        return nll, {
            "nll": nll.detach(),
            "integral": (integral / denominator).detach(),
            "negative_log_term": (-log_term / denominator).detach(),
        }

    def type_regularization(self, alpha, target_type_ids, history_type_ids, pair_mask):
        selected_alpha = alpha.gather(
            -1,
            target_type_ids[:, None, None].expand(-1, alpha.shape[1], 1),
        ).squeeze(-1)
        source_types = history_type_ids
        target_types = target_type_ids[:, None].expand_as(selected_alpha)
        pair_ids = target_types * self.config.n_types + source_types
        valid_pair_ids = pair_ids.masked_select(pair_mask)
        valid_scores = selected_alpha.masked_select(pair_mask)
        if valid_scores.numel() == 0:
            zero = alpha.sum() * 0.0
            return zero, zero

        size = self.config.n_types * self.config.n_types
        score_sums = torch.zeros(size, dtype=alpha.dtype, device=alpha.device)
        score_square_sums = torch.zeros_like(score_sums)
        counts = torch.zeros_like(score_sums)
        score_sums.scatter_add_(0, valid_pair_ids, valid_scores)
        score_square_sums.scatter_add_(0, valid_pair_ids, valid_scores * valid_scores)
        counts.scatter_add_(0, valid_pair_ids, torch.ones_like(valid_scores))

        observed = counts > 0
        means = score_sums[observed] / counts[observed]
        l1_mean = means.abs().sum()
        variances = score_square_sums[observed] / counts[observed] - means * means
        variance = variances.clamp_min(0.0).sum()
        return l1_mean, variance

    def accumulate_type_scores(self, batch, score_sums=None, counts=None):
        _intensities, _mu, alpha, _gamma, pair_mask = self.intensity_at_events(
            batch["target_times"],
            batch["query_dts"],
            batch["query_alarm_source_ids"],
            batch["query_alarm_type_ids"],
            batch["history_times"],
            batch["history_dts"],
            batch["history_alarm_source_ids"],
            batch["history_alarm_type_ids"],
            batch["history_mask"],
            topology_pair_features=batch["topology_pair_features"],
        )
        selected_alpha = alpha.gather(
            -1,
            batch["target_type_ids"][:, None, None].expand(-1, alpha.shape[1], 1),
        ).squeeze(-1)
        source_types = batch["history_type_ids"]
        target_types = batch["target_type_ids"][:, None].expand_as(selected_alpha)
        pair_ids = target_types * self.config.n_types + source_types
        valid_pair_ids = pair_ids.masked_select(pair_mask)
        valid_scores = selected_alpha.masked_select(pair_mask)
        size = self.config.n_types * self.config.n_types
        if score_sums is None:
            score_sums = torch.zeros(size, dtype=alpha.dtype, device=alpha.device)
            counts = torch.zeros_like(score_sums)
        score_sums.scatter_add_(0, valid_pair_ids, valid_scores)
        counts.scatter_add_(0, valid_pair_ids, torch.ones_like(valid_scores))
        return score_sums, counts


def move_batch_to_device(batch, device):
    require_torch()
    moved_batch = {**batch}
    tensor_names = (
        "target_type_ids",
        "target_times",
        "interval_dts",
        "query_dts",
        "query_alarm_source_ids",
        "query_alarm_type_ids",
        "history_times",
        "history_dts",
        "history_type_ids",
        "history_alarm_source_ids",
        "history_alarm_type_ids",
        "history_mask",
        "history_lengths",
    )
    for name in tensor_names:
        moved_batch[name] = batch[name].to(device)
    if batch["topology_pair_features"] is not None:
        moved_batch["topology_pair_features"] = batch["topology_pair_features"].to(device)
    return moved_batch


def average_type_score_matrix(model, dataloader, device):
    require_torch()
    model.eval()
    score_sums = None
    counts = None
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            score_sums, counts = model.accumulate_type_scores(batch, score_sums, counts)
    if score_sums is None:
        empty = torch.zeros(model.config.n_types, model.config.n_types)
        return empty, empty
    score_matrix = (score_sums / counts.clamp_min(1.0)).reshape(model.config.n_types, model.config.n_types)
    count_matrix = counts.reshape(model.config.n_types, model.config.n_types)
    return score_matrix.detach().cpu(), count_matrix.detach().cpu()


def save_alarm_isahp_artifact(path, model, vocabs, sequence_config, training_payload):
    require_torch()
    torch.save(
        {
            "artifact_type": "alarm_flow_isahp.v7",
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "vocabs": vocabs.to_dict(),
            "sequence_config": sequence_config.to_dict(),
            "training": dict(training_payload or {}),
        },
        path,
    )


def load_alarm_isahp_artifact(path, *, map_location="cpu"):
    require_torch()
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if payload.get("artifact_type") != "alarm_flow_isahp.v7":
        raise ValueError(f"unsupported alarm ISAHP artifact: {payload.get('artifact_type')}")
    model = AlarmFlowISAHP(AlarmISAHPConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model, payload


def summarize_alarm_event(item):
    alarm = item.get("alarm", {})
    return {
        "event_id": alarm.get("告警编码ID", ""),
        "site_id": item.get("site_id", ""),
        "alarm_source": item.get("alarm_source", ""),
        "alarm_title": item.get("alarm_title", ""),
        "alarm_type": alarm_type_from_title(item.get("alarm_title", "")),
        "ts": float(item["ts"]),
        "is_clear": is_clear_alarm(alarm),
    }


def iter_instance_edges(model, sequences, device, *, top_parents=5):
    require_torch()
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            if not sequence.target_windows:
                continue
            batch = move_batch_to_device(
                collate_alarm_target_windows(sequence.target_windows),
                device,
            )
            _intensities, _mu, alpha, gamma, pair_mask = model.intensity_at_events(
                batch["target_times"],
                batch["query_dts"],
                batch["query_alarm_source_ids"],
                batch["query_alarm_type_ids"],
                batch["history_times"],
                batch["history_dts"],
                batch["history_alarm_source_ids"],
                batch["history_alarm_type_ids"],
                batch["history_mask"],
                topology_pair_features=batch["topology_pair_features"],
            )
            alpha = alpha.detach().cpu()
            gamma = gamma.detach().cpu()
            pair_mask = pair_mask.detach().cpu()
            target_times = batch["target_times"].detach().cpu()
            history_times = batch["history_times"].detach().cpu()
            for row, window in enumerate(sequence.target_windows):
                target_type = window.target_type_id
                scores = alpha[row, : len(window), target_type]
                decay_rates = gamma[row, : len(window), target_type]
                target_ts = target_times[row]
                parent_times = history_times[row, : len(window)]
                contributions = scores * decay_rates * torch.exp(
                    -decay_rates * (target_ts - parent_times).clamp_min(0.0)
                )
                parent_offsets = [
                    source_offset
                    for source_offset in range(len(window))
                    if bool(pair_mask[row, source_offset])
                ]
                parent_offsets.sort(key=lambda index: float(scores[index]), reverse=True)
                if top_parents > 0:
                    parent_offsets = parent_offsets[:top_parents]
                for source_offset in parent_offsets:
                    edge = {
                        "sequence_id": sequence.sequence_id,
                        "source_index": window.history_indices[source_offset],
                        "target_index": window.target_index,
                        "source_type": window.history_type_labels[source_offset],
                        "target_type": window.target_type_label,
                        "alpha": float(scores[source_offset]),
                        "decay_rate": float(decay_rates[source_offset]),
                        "contribution_at_target": float(contributions[source_offset]),
                        "source_event": summarize_alarm_event(window.history_events[source_offset]),
                        "target_event": summarize_alarm_event(window.target_event),
                    }
                    if window.topology_pair_features:
                        edge["ne_topology_features"] = {
                            feature_name: float(feature_value)
                            for feature_name, feature_value in zip(
                                PAIR_FEATURE_NAMES,
                                window.topology_pair_features[source_offset],
                            )
                        }
                    yield edge
