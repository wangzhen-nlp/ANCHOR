#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""site_chains 站点上下游/平行关系查询界面。"""

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
    TK_IMPORT_ERROR = exc

    class _MissingTtk:
        class Frame:
            pass

    ttk = _MissingTtk()

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.resource_buffer import load_resource_buffer
from anchor_grouping_online.site_topology import build_site_chain_index
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL


DEFAULT_RESOURCE_BUFFER = Path(RESOURCE_BUFFER_JSONL)
RELATION_LABELS = {
    "downstream": "下游",
    "upstream": "上游",
    "parallel": "平行",
}


def normalize_site_key(value) -> str:
    return str(value or "").strip().upper()


def load_site_chain_index(resource_buffer_path: str):
    resources = load_resource_buffer(resource_buffer_path, ("site_chains",))
    site_chains = resources["site_chains"]
    index = build_site_chain_index(site_chains)
    meta = site_chains.get("meta", {}) if isinstance(site_chains, dict) else {}
    lookup = {normalize_site_key(site_id): site_id for site_id in index}
    return index, lookup, meta


def related_rows(site_index: dict, site_id: str):
    info = site_index.get(site_id)
    if not info:
        return {
            "downstream": [],
            "upstream": [],
            "parallel": [],
        }

    downstream = sorted(
        ((int(hop), related_site) for related_site, hop in info["downstream_site_hops"].items()),
        key=lambda item: (item[0], item[1]),
    )
    upstream = sorted(
        ((int(hop), related_site) for related_site, hop in info["upstream_site_hops"].items()),
        key=lambda item: (item[0], item[1]),
    )
    parallel = [(1, related_site) for related_site in sorted(info["bidirectional_sites"])]
    return {
        "downstream": downstream,
        "upstream": upstream,
        "parallel": parallel,
    }


def find_site(site_lookup: dict, raw_site_id: str):
    key = normalize_site_key(raw_site_id)
    if not key:
        return None, []
    exact = site_lookup.get(key)
    if exact:
        return exact, []

    matches = [
        site_id
        for lookup_key, site_id in site_lookup.items()
        if key in lookup_key or key in normalize_site_key(site_id)
    ]
    matches = sorted(set(matches))[:20]
    return None, matches


def meta_summary(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    return (
        f"站点数: {meta.get('site_count', '-')}; "
        f"下游关系: {meta.get('total_downstream_relations', '-')}; "
        f"平行边: {meta.get('total_bidirectional_edges', '-')}"
    )


class RelationTable(ttk.Frame):
    def __init__(self, parent, relation_key: str, on_site_open=None):
        super().__init__(parent)
        self.relation_key = relation_key
        self.on_site_open = on_site_open
        self.count_var = tk.StringVar(value="0 个站点")

        header = ttk.Frame(self)
        header.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(header, text=RELATION_LABELS[relation_key], font=("", 11, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.count_var).pack(side="right")

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tree = ttk.Treeview(
            table_frame,
            columns=("hop", "site_id"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("hop", text="跳数")
        self.tree.heading("site_id", text="站点")
        self.tree.column("hop", width=70, minwidth=60, anchor="center", stretch=False)
        self.tree.column("site_id", width=360, minwidth=180, anchor="w", stretch=True)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._on_double_click)

    def set_rows(self, rows):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for hop, site_id in rows:
            hop_text = str(hop) if self.relation_key != "parallel" else "-"
            self.tree.insert("", "end", values=(hop_text, site_id))
        self.count_var.set(f"{len(rows)} 个站点")

    def _on_double_click(self, event):
        if not self.on_site_open:
            return
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        values = self.tree.item(item_id, "values")
        if len(values) < 2:
            return
        site_id = str(values[1]).strip()
        if site_id:
            self.on_site_open(site_id)


class App:
    def __init__(self, root: tk.Tk, resource_path: Path):
        self.root = root
        self.root.title("site_chains 站点关系查询")
        self.root.minsize(720, 520)

        self.site_index = {}
        self.site_lookup = {}
        self.loaded_path = None
        self.load_queue = queue.Queue()
        self.loading = False

        self.resource_var = tk.StringVar(value=str(resource_path))
        self.site_var = tk.StringVar()
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

        ttk.Label(outer, text="资源文件").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        self.resource_entry = ttk.Entry(outer, textvariable=self.resource_var)
        self.resource_entry.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(outer, text="浏览", command=self._browse_resource).grid(
            row=0, column=2, sticky="e", padx=(8, 0), pady=(0, 6)
        )
        self.load_button = ttk.Button(outer, text="加载", command=self._start_load)
        self.load_button.grid(row=0, column=3, sticky="e", padx=(8, 0), pady=(0, 6))

        ttk.Label(outer, text="站点").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        self.site_entry = ttk.Entry(outer, textvariable=self.site_var)
        self.site_entry.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        self.site_entry.bind("<Return>", lambda _event: self._query())
        self.query_button = ttk.Button(outer, text="查询", command=self._query)
        self.query_button.grid(row=1, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(0, 6))

        ttk.Label(outer, textvariable=self.status_var, foreground="#555555").grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(2, 0)
        )
        ttk.Label(outer, textvariable=self.summary_var, foreground="#1f6f3f").grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(6, 6)
        )

        self.notebook = ttk.Notebook(outer)
        self.notebook.grid(row=4, column=0, columnspan=4, sticky="nsew", pady=(2, 0))
        self.tables = {}
        for relation_key in ("downstream", "upstream", "parallel"):
            table = RelationTable(self.notebook, relation_key, on_site_open=self._open_related_site)
            self.tables[relation_key] = table
            self.notebook.add(table, text=RELATION_LABELS[relation_key])

    def _set_loaded(self, loaded: bool):
        state = "normal" if loaded and not self.loading else "disabled"
        self.query_button.configure(state=state)
        self.site_entry.configure(state=state)

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
        self.status_var.set(f"正在加载 site_chains: {path}")
        self.summary_var.set("")
        self._clear_tables()

        thread = threading.Thread(target=self._load_worker, args=(path,), daemon=True)
        thread.start()
        self.root.after(100, self._poll_load_queue)

    def _load_worker(self, path: Path):
        started = time.time()
        try:
            site_index, site_lookup, meta = load_site_chain_index(str(path))
        except BaseException:
            self.load_queue.put(("error", path, traceback.format_exc()))
            return
        self.load_queue.put(("ok", path, site_index, site_lookup, meta, time.time() - started))

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
            self.site_index = {}
            self.site_lookup = {}
            self.loaded_path = None
            self._set_loaded(False)
            self.status_var.set(f"加载失败: {path}")
            messagebox.showerror("加载失败", details.splitlines()[-1] if details else "未知错误")
            return

        _kind, path, site_index, site_lookup, meta, elapsed = item
        self.site_index = site_index
        self.site_lookup = site_lookup
        self.loaded_path = path
        self._set_loaded(True)
        self.status_var.set(f"已加载: {path} ({elapsed:.1f}s)")
        self.summary_var.set(meta_summary(meta))
        self.site_entry.focus_set()

    def _query(self):
        raw_site_id = self.site_var.get().strip()
        if not raw_site_id:
            messagebox.showwarning("提示", "请输入站点 ID")
            return
        if not self.site_index:
            messagebox.showwarning("提示", "请先加载资源文件")
            return

        site_id, matches = find_site(self.site_lookup, raw_site_id)
        if not site_id:
            self._clear_tables()
            if matches:
                self.status_var.set(
                    "未找到精确站点。可能匹配: " + ", ".join(matches)
                )
            else:
                self.status_var.set(f"未找到站点: {raw_site_id}")
            return

        rows = related_rows(self.site_index, site_id)
        for relation_key, table in self.tables.items():
            table.set_rows(rows[relation_key])
        self.status_var.set(f"当前站点: {site_id}")
        self.summary_var.set(
            f"下游 {len(rows['downstream'])} 个; "
            f"上游 {len(rows['upstream'])} 个; "
            f"平行 {len(rows['parallel'])} 个"
        )

    def _open_related_site(self, site_id: str):
        self.site_var.set(site_id)
        self._query()
        self.site_entry.focus_set()

    def _clear_tables(self):
        for table in self.tables.values():
            table.set_rows([])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="查询 resource_buffer.jsonl 中的 site_chains 站点关系")
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
