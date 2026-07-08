#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_and_find_neighbors.py 的图形界面版本：
在两个文本框中分别输入站点字符串和输出文件名，点击"生成"按钮生成结果文件
"""

import json
import sys
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, scrolledtext

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_site_ids import extract_site_ids
from find_site_neighbors import build_site_graph, get_nth_order_neighbors
from filter_links import get_nes_by_site_ids

DEFAULT_GRAPH_FILE = REPO_ROOT / "topology_resources" / "ne_graph.json"
DEFAULT_N = 1


def run_pipeline(text: str, output_path: Path, graph_file: str, n: int) -> str:
    start_sites = [s.strip() for s in extract_site_ids(text).split(',') if s.strip()]
    if not start_sites:
        raise ValueError("未从输入中抽取到任何 site_id")

    site_graph = build_site_graph(graph_file)
    neighbor_sites = get_nth_order_neighbors(start_sites, n, site_graph)
    result = get_nes_by_site_ids(graph_file, neighbor_sites)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return (f"已保存到: {output_path}\n"
            f"起始站点数: {len(start_sites)}\n"
            f"{n} 阶邻居总站点数: {len(neighbor_sites)}\n"
            f"总 NE 数: {len(result['ne_info'])}\n"
            f"总站点数: {len(result['group_info']['1']['site_list'])}")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("站点邻居生成工具")

        tk.Label(root, text="站点字符串（extract_site_ids 的输入）:").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        self.text_input = scrolledtext.ScrolledText(root, width=72, height=12)
        self.text_input.grid(row=1, column=0, columnspan=2, padx=8, pady=4, sticky="nsew")

        tk.Label(root, text="输出文件名:").grid(row=2, column=0, sticky="w", padx=8)
        self.output_entry = tk.Entry(root, width=60)
        self.output_entry.insert(0, "output.json")
        self.output_entry.grid(row=3, column=0, columnspan=2, padx=8, pady=4, sticky="we")

        tk.Label(root, text="ne_graph.json 路径:").grid(row=4, column=0, sticky="w", padx=8)
        self.graph_entry = tk.Entry(root, width=60)
        self.graph_entry.insert(0, str(DEFAULT_GRAPH_FILE))
        self.graph_entry.grid(row=5, column=0, columnspan=2, padx=8, pady=4, sticky="we")

        tk.Label(root, text="阶数 n:").grid(row=6, column=0, sticky="w", padx=8)
        self.n_entry = tk.Entry(root, width=6)
        self.n_entry.insert(0, str(DEFAULT_N))
        self.n_entry.grid(row=6, column=0, padx=(70, 0), sticky="w")

        self.generate_btn = tk.Button(root, text="生成", width=16, command=self.on_generate)
        self.generate_btn.grid(row=7, column=0, columnspan=2, pady=8)

        self.status = tk.Label(root, text="", justify="left", anchor="w", fg="gray")
        self.status.grid(row=8, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

    def on_generate(self):
        text = self.text_input.get("1.0", tk.END)
        output_name = self.output_entry.get().strip()
        graph_file = self.graph_entry.get().strip()

        if not output_name:
            messagebox.showwarning("提示", "请填写输出文件名")
            return
        if not output_name.lower().endswith('.json'):
            output_name += '.json'
        try:
            n = int(self.n_entry.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "阶数 n 必须是整数")
            return

        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

        try:
            summary = run_pipeline(text, output_path, graph_file, n)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("生成失败", str(e))
            self.status.config(text=f"失败: {e}", fg="red")
            return

        self.status.config(text=summary, fg="green")
        messagebox.showinfo("完成", summary)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
