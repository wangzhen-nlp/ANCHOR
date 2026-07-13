#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输入 NE ID 查询其所属站点信息的图形界面。

数据来自 build_resource_buffer.py 生成的 resource_buffer.jsonl：
- ne_graph:           NE 基础信息与所属站点（site_id/site_name/经纬度）
- site_graph:         站点补充信息（is_hub）
- site_device_counts: 站点各 domain 的设备数量
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import traceback
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    TK_IMPORT_ERROR = None
except ImportError as exc:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
    TK_IMPORT_ERROR = exc

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root()

from anchor_grouping_online.resource_buffer import load_resource_buffer
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL


DEFAULT_RESOURCE_BUFFER = Path(RESOURCE_BUFFER_JSONL)

# ne_graph 中保留展示的 NE 字段（其余字段与 link 邻接一并丢弃以省内存）
_NE_FIELDS = ("name", "domain", "type", "network_type", "manufacturer", "site_id")

# 模糊匹配时最多提示的候选数
_MAX_FUZZY_MATCHES = 20


def normalize_key(value) -> str:
    return str(value or "").strip().upper()


def load_ne_site_index(resource_buffer_path: str):
    """加载并瘦身缓冲资源，返回查询所需索引。

    Returns:
        (ne_index, ne_lookup, site_info, site_device_counts, site_nes)
        - ne_index:   {ne_id: {name, domain, type, network_type, manufacturer, site_id}}
        - ne_lookup:  {规范化 ne_id: ne_id}
        - site_info:  {site_id: {site_name, longitude, latitude, is_hub}}
        - site_device_counts: {site_id: {domain: count}}
        - site_nes:   {site_id: [ne_id, ...]}
    """
    resources = load_resource_buffer(
        resource_buffer_path, ("site_graph", "ne_graph", "site_device_counts")
    )

    site_info = {}
    for site_id, site_data in resources["site_graph"].items():
        site_info[site_id] = {
            "site_name": site_data.get("site_name", ""),
            "longitude": site_data.get("longitude", ""),
            "latitude": site_data.get("latitude", ""),
            "is_hub": bool(site_data.get("is_hub", False)),
        }

    ne_index = {}
    site_nes = {}
    for ne_id, ne_data in resources["ne_graph"].items():
        ne_index[ne_id] = {field: ne_data.get(field, "") for field in _NE_FIELDS}
        site_id = ne_data.get("site_id", "")
        if site_id:
            site_nes.setdefault(site_id, []).append(ne_id)
            # ne_graph 中冗余的站点名/经纬度用于兜底 site_graph 缺失的站点
            if site_id not in site_info:
                site_info[site_id] = {
                    "site_name": ne_data.get("site_name", ""),
                    "longitude": ne_data.get("longitude", ""),
                    "latitude": ne_data.get("latitude", ""),
                    "is_hub": False,
                }

    ne_lookup = {normalize_key(ne_id): ne_id for ne_id in ne_index}
    return ne_index, ne_lookup, site_info, resources["site_device_counts"], site_nes


def find_ne(ne_lookup: dict, raw_ne_id: str):
    """精确（大小写不敏感）查找 NE；找不到时返回子串模糊候选。"""
    key = normalize_key(raw_ne_id)
    if not key:
        return None, []
    exact = ne_lookup.get(key)
    if exact:
        return exact, []

    matches = sorted(
        ne_id for lookup_key, ne_id in ne_lookup.items() if key in lookup_key
    )[:_MAX_FUZZY_MATCHES]
    return None, matches


def format_device_counts(device_counts: dict) -> str:
    if not device_counts:
        return "-"
    items = sorted(device_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{domain}: {count}" for domain, count in items)


def build_result_text(ne_id: str, ne_data: dict, site_info: dict,
                      site_device_counts: dict) -> str:
    """拼出 NE 与站点信息的文本报告。"""
    lines = [
        "[NE 信息]",
        f"  NE ID:        {ne_id}",
        f"  名称:         {ne_data.get('name') or '-'}",
        f"  domain:       {ne_data.get('domain') or '-'}",
        f"  类型:         {ne_data.get('type') or '-'}",
        f"  网络类型:     {ne_data.get('network_type') or '-'}",
        f"  厂商:         {ne_data.get('manufacturer') or '-'}",
        "",
        "[站点信息]",
    ]

    site_id = ne_data.get("site_id", "")
    if not site_id:
        lines.append("  该 NE 未关联任何站点（site_id 为空）")
        return "\n".join(lines)

    site_data = site_info.get(site_id, {})
    lines.extend([
        f"  站点 ID:      {site_id}",
        f"  站点名称:     {site_data.get('site_name') or '-'}",
        f"  经度:         {site_data.get('longitude') or '-'}",
        f"  纬度:         {site_data.get('latitude') or '-'}",
        f"  是否枢纽站:   {'是' if site_data.get('is_hub') else '否'}",
        f"  设备画像:     {format_device_counts(site_device_counts.get(site_id, {}))}",
    ])
    return "\n".join(lines)


class App:
    def __init__(self, root: tk.Tk, resource_path: Path):
        self.root = root
        self.root.title("NE 站点信息查询")
        self.root.minsize(760, 560)

        self.ne_index = {}
        self.ne_lookup = {}
        self.site_info = {}
        self.site_device_counts = {}
        self.site_nes = {}
        self.loaded_path = None
        self.load_queue = queue.Queue()
        self.loading = False

        self.resource_var = tk.StringVar(value=str(resource_path))
        self.ne_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择 resource_buffer.jsonl 并加载")
        self.summary_var = tk.StringVar(value="")

        self._build_layout()
        self._set_loaded(False)
        if resource_path.exists():
            self._start_load()

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(4, weight=1)
        outer.rowconfigure(5, weight=2)

        ttk.Label(outer, text="资源文件").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        self.resource_entry = ttk.Entry(outer, textvariable=self.resource_var)
        self.resource_entry.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(outer, text="浏览", command=self._browse_resource).grid(
            row=0, column=2, sticky="e", padx=(8, 0), pady=(0, 6)
        )
        self.load_button = ttk.Button(outer, text="加载", command=self._start_load)
        self.load_button.grid(row=0, column=3, sticky="e", padx=(8, 0), pady=(0, 6))

        ttk.Label(outer, text="NE ID").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        self.ne_entry = ttk.Entry(outer, textvariable=self.ne_var)
        self.ne_entry.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        self.ne_entry.bind("<Return>", lambda _event: self._query())
        self.query_button = ttk.Button(outer, text="查询", command=self._query)
        self.query_button.grid(row=1, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(0, 6))

        ttk.Label(outer, textvariable=self.status_var, foreground="#555555").grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(2, 0)
        )
        ttk.Label(outer, textvariable=self.summary_var, foreground="#1f6f3f").grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(6, 6)
        )

        # 结果文本：NE 信息 + 站点信息
        result_frame = ttk.Frame(outer)
        result_frame.grid(row=4, column=0, columnspan=4, sticky="nsew", pady=(2, 6))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self.result_text = tk.Text(result_frame, height=13, wrap="none", state="disabled")
        self.result_text.grid(row=0, column=0, sticky="nsew")
        result_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        result_scroll.grid(row=0, column=1, sticky="ns")

        # 同站点 NE 列表
        table_outer = ttk.Frame(outer)
        table_outer.grid(row=5, column=0, columnspan=4, sticky="nsew")
        table_outer.columnconfigure(0, weight=1)
        table_outer.rowconfigure(1, weight=1)

        header = ttk.Frame(table_outer)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(header, text="同站点 NE（双击查询）", font=("", 11, "bold")).pack(side="left")
        self.site_ne_count_var = tk.StringVar(value="0 个 NE")
        ttk.Label(header, textvariable=self.site_ne_count_var).pack(side="right")

        self.tree = ttk.Treeview(
            table_outer,
            columns=("ne_id", "name", "domain", "type"),
            show="headings",
            selectmode="browse",
        )
        for column, text, width, stretch in (
            ("ne_id", "NE ID", 220, True),
            ("name", "名称", 220, True),
            ("domain", "domain", 120, False),
            ("type", "类型", 120, False),
        ):
            self.tree.heading(column, text=text)
            self.tree.column(column, width=width, minwidth=80, anchor="w", stretch=stretch)
        self.tree.grid(row=1, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(table_outer, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=1, column=1, sticky="ns")
        self.tree.bind("<Double-1>", self._on_tree_double_click)

    def _set_loaded(self, loaded: bool):
        state = "normal" if loaded and not self.loading else "disabled"
        self.query_button.configure(state=state)
        self.ne_entry.configure(state=state)

    def _browse_resource(self):
        path = filedialog.askopenfilename(
            title="选择 build_resource_buffer.py 生成的 JSONL 文件",
            initialdir=str(DEFAULT_RESOURCE_BUFFER.parent),
            filetypes=(("JSONL 文件", "*.jsonl"), ("所有文件", "*.*")),
        )
        if path:
            self.resource_var.set(path)

    def _start_load(self):
        if self.loading:
            return
        path = Path(self.resource_var.get().strip())
        if not path:
            messagebox.showwarning("提示", "请先选择资源文件")
            return
        if not path.exists():
            messagebox.showerror("加载失败", f"文件不存在: {path}")
            return

        self.loading = True
        self.load_button.configure(state="disabled")
        self._set_loaded(False)
        self.status_var.set(f"正在加载 ne_graph/site_graph: {path}")
        self.summary_var.set("")
        self._clear_result()

        thread = threading.Thread(target=self._load_worker, args=(path,), daemon=True)
        thread.start()
        self.root.after(100, self._poll_load_queue)

    def _load_worker(self, path: Path):
        started = time.time()
        try:
            loaded = load_ne_site_index(str(path))
        except BaseException:
            self.load_queue.put(("error", path, traceback.format_exc()))
            return
        self.load_queue.put(("ok", path, loaded, time.time() - started))

    def _poll_load_queue(self):
        try:
            item = self.load_queue.get_nowait()
        except queue.Empty:
            if self.loading:
                self.root.after(100, self._poll_load_queue)
            return

        kind = item[0]
        self.loading = False
        self.load_button.configure(state="normal")
        if kind == "error":
            _kind, path, details = item
            self.ne_index = {}
            self.ne_lookup = {}
            self.site_info = {}
            self.site_device_counts = {}
            self.site_nes = {}
            self.loaded_path = None
            self._set_loaded(False)
            self.status_var.set(f"加载失败: {path}")
            messagebox.showerror("加载失败", details.splitlines()[-1] if details else "未知错误")
            return

        _kind, path, loaded, elapsed = item
        (self.ne_index, self.ne_lookup, self.site_info,
         self.site_device_counts, self.site_nes) = loaded
        self.loaded_path = path
        self._set_loaded(True)
        self.status_var.set(f"已加载: {path} ({elapsed:.1f}s)")
        self.summary_var.set(
            f"NE 数: {len(self.ne_index)}; 站点数: {len(self.site_info)}"
        )
        self.ne_entry.focus_set()

    def _query(self):
        raw_ne_id = self.ne_var.get().strip()
        if not raw_ne_id:
            messagebox.showwarning("提示", "请输入 NE ID")
            return
        if not self.ne_index:
            messagebox.showwarning("提示", "请先加载资源文件")
            return

        ne_id, matches = find_ne(self.ne_lookup, raw_ne_id)
        if not ne_id:
            self._clear_result()
            if matches:
                self.status_var.set("未找到精确 NE。可能匹配: " + ", ".join(matches))
            else:
                self.status_var.set(f"未找到 NE: {raw_ne_id}")
            return

        ne_data = self.ne_index[ne_id]
        site_id = ne_data.get("site_id", "")
        self._set_result_text(
            build_result_text(ne_id, ne_data, self.site_info, self.site_device_counts)
        )

        same_site_nes = [
            other for other in self.site_nes.get(site_id, []) if other != ne_id
        ] if site_id else []
        self._set_site_ne_rows(same_site_nes)

        self.status_var.set(f"当前 NE: {ne_id}")
        if site_id:
            self.summary_var.set(
                f"所属站点: {site_id}; 同站点其它 NE: {len(same_site_nes)} 个"
            )
        else:
            self.summary_var.set("该 NE 未关联站点")

    def _set_result_text(self, text: str):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _set_site_ne_rows(self, ne_ids):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for other_id in sorted(ne_ids):
            other = self.ne_index.get(other_id, {})
            self.tree.insert("", "end", values=(
                other_id,
                other.get("name", ""),
                other.get("domain", ""),
                other.get("type", ""),
            ))
        self.site_ne_count_var.set(f"{len(ne_ids)} 个 NE")

    def _on_tree_double_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        values = self.tree.item(item_id, "values")
        if not values:
            return
        ne_id = str(values[0]).strip()
        if ne_id:
            self.ne_var.set(ne_id)
            self._query()
            self.ne_entry.focus_set()

    def _clear_result(self):
        self._set_result_text("")
        self._set_site_ne_rows([])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="根据 NE ID 查询 resource_buffer.jsonl 中的所属站点信息"
    )
    parser.add_argument(
        "--resource-buffer",
        default=str(DEFAULT_RESOURCE_BUFFER),
        help=f"build_resource_buffer.py 生成的 JSONL 文件，默认: {DEFAULT_RESOURCE_BUFFER}",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if TK_IMPORT_ERROR is not None:
        raise SystemExit(
            "当前 Python 未启用 tkinter，无法启动图形界面；"
            "请在 Windows 上使用带 Tcl/Tk 的标准 Python 运行。"
        ) from TK_IMPORT_ERROR
    root = tk.Tk()
    App(root, Path(args.resource_buffer))
    root.mainloop()


if __name__ == "__main__":
    main()
