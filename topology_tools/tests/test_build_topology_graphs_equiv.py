#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
等价性测试：保证 build_topology_graphs 的三个产物与「分别执行三个独立工具」逐一致。

做法：
1. 在临时目录生成覆盖各类边界的合成数据；
2. 用 subprocess 分别执行
   - extract_site_graph      -> site_graph.json
   - extract_ne_graph        -> ne_graph.json（依赖上一步产物）
   - build_link_peer_index   -> link_peer_index.json
   以及 build_topology_graphs 一次性生成三者；
3. 以「顺序无关」方式比较 JSON 内容（顶层 key 顺序受进程哈希随机化影响，原工具间本身也不稳定）。

覆盖的边界：
- NE / 站点 / 链路 三类的 last_Modified 去重取最新
- 反向链路升级为 '<->'
- NE 无站点（site 图跳过、ne 图保留、region 来自 NE）
- 链路缺端口（peer_index 跳过、两张图仍登记）
- NE 引用了 SYS_SITE 中不存在的站点（site 图使用默认站点）
- a_end_ne_nativeId(') 变体列
- 无去重键的链路（原样保留、不去重）+ 小写 NE 归一化
- 缺一端 NE 的链路（全部跳过）
- CSV / zip(内含CSV) / JSONL 混合输入，链路以目录形式传入
- ne_graph 的 region_id 从站点回填
"""

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #
NE_ROWS = [
    dict(nativeId="NE1", ne_site_id="sA", domain="D1", typeId="T1", network_type="N1",
         name="ne1-old", manufacturer="M", regionId1="R1", running_status="up", last_Modified="1000"),
    # NE1 重复且更新 -> 取最新（name 变化用于验证 latest-wins）
    dict(nativeId="NE1", ne_site_id="sA", domain="D1", typeId="T1", network_type="N1",
         name="ne1-new", manufacturer="M", regionId1="R1", running_status="up", last_Modified="2000"),
    dict(nativeId="NE2", ne_site_id="sB", domain="D2", typeId="T2", network_type="N1",
         name="ne2", manufacturer="M", regionId1="", running_status="up", last_Modified="1000"),
    dict(nativeId="NE3", ne_site_id="sB", domain="D3", typeId="T3", network_type="N2",
         name="ne3", manufacturer="M", regionId1="R3", running_status="down", last_Modified="1000"),
    # NE4 无站点 -> ne_site_map 不含；ne 图保留，region 来自 NE
    dict(nativeId="NE4", ne_site_id="", domain="D4", typeId="T4", network_type="N2",
         name="ne4", manufacturer="M", regionId1="R4", running_status="up", last_Modified="1000"),
    # NE5 引用 sZ，但 SYS_SITE 无 SZ -> site 图用默认站点
    dict(nativeId="NE5", ne_site_id="sZ", domain="D5", typeId="T5", network_type="N1",
         name="ne5", manufacturer="M", regionId1="", running_status="up", last_Modified="1000"),
    # NE6 有站点但无任何链路
    dict(nativeId="NE6", ne_site_id="sB", domain="D6", typeId="T6", network_type="N1",
         name="ne6", manufacturer="M", regionId1="R3", running_status="up", last_Modified="1000"),
]

SITE_ROWS = [
    dict(site_id="SA", name="siteA", site_type="core", longitude="1.0", latitude="2.0",
         region_id="R1", is_hub="true", last_Modified="1000"),
    dict(site_id="SB", name="siteB-old", site_type="edge", longitude="3.0", latitude="4.0",
         region_id="R3", is_hub="0", last_Modified="1000"),
    # SB 重复且更新 -> 取最新
    dict(site_id="SB", name="siteB-new", site_type="edge", longitude="3.0", latitude="4.0",
         region_id="R3", is_hub="0", last_Modified="2000"),
    # SC 无 NE 无链路 -> 仅出现在 site_info，link 为空
    dict(site_id="SC", name="siteC", site_type="edge", longitude="5.0", latitude="6.0",
         region_id="R9", is_hub="false", last_Modified="1000"),
]

# 链路：分散到 jsonl / csv / zip 三个文件，统一放入一个目录
LINKS_JSONL = [
    dict(nativeId="L1", a_end_ne_nativeId="ne1", z_end_ne_nativeId="ne2", link_layer="OTN",
         a_end_port_name="p1", z_end_port_name="p2", a_end_port_ip="10.0.0.1", z_end_port_ip="10.0.0.2",
         a_end_ne_manager_name="mgrA-old", z_end_ne_manager_name="mgrB", last_Modified="1000"),
    # L1 重复且更新 -> 取最新（manager 变化）
    dict(nativeId="L1", a_end_ne_nativeId="ne1", z_end_ne_nativeId="ne2", link_layer="OTN",
         a_end_port_name="p1", z_end_port_name="p2", a_end_port_ip="10.0.0.9", z_end_port_ip="10.0.0.2",
         a_end_ne_manager_name="mgrA-new", z_end_ne_manager_name="mgrB", last_Modified="3000"),
    # L2 反向 ne2->ne1 OTN -> 与 L1 合成 '<->'
    dict(nativeId="L2", a_end_ne_nativeId="ne2", z_end_ne_nativeId="ne1", link_layer="OTN",
         a_end_port_name="p3", z_end_port_name="p4", a_end_port_ip="", z_end_port_ip="",
         a_end_ne_manager_name="", z_end_ne_manager_name="", last_Modified="1000"),
]
LINKS_CSV = [
    # L3 缺端口 -> peer_index 跳过；两张图仍登记
    dict(nativeId="L3", a_end_ne_nativeId="ne2", z_end_ne_nativeId="ne3", link_layer="IP",
         a_end_port_name="", z_end_port_name="", a_end_port_ip="", z_end_port_ip="",
         a_end_ne_manager_name="", z_end_ne_manager_name="", last_Modified="1000"),
    # L4 ne4 无站点 -> site 图跳过；ne 图保留
    dict(nativeId="L4", a_end_ne_nativeId="ne4", z_end_ne_nativeId="ne1", link_layer="IP",
         a_end_port_name="p7", z_end_port_name="p8", a_end_port_ip="", z_end_port_ip="",
         a_end_ne_manager_name="", z_end_ne_manager_name="", last_Modified="1000"),
    # 缺一端 NE -> 全部跳过
    dict(nativeId="L8", a_end_ne_nativeId="ne1", z_end_ne_nativeId="", link_layer="OTN",
         a_end_port_name="p9", z_end_port_name="pa", a_end_port_ip="", z_end_port_ip="",
         a_end_ne_manager_name="", z_end_ne_manager_name="", last_Modified="1000"),
]
# zip 内 csv：含变体列与无 key 记录
LINKS_ZIP = [
    # L5 ne5 -> sZ（SYS_SITE 无 SZ）；site 图用默认站点
    dict(nativeId="L5", a_end_ne_nativeId="ne5", z_end_ne_nativeId="ne1", link_layer="OTN",
         a_end_port_name="pz1", z_end_port_name="pz2", a_end_port_ip="", z_end_port_ip="",
         a_end_ne_manager_name="", z_end_ne_manager_name="", last_Modified="1000"),
]
LINKS_ZIP_VARIANT = [
    # 变体列名 a_end_ne_nativeId(') / z_end_ne_nativeId(')，并用小写验证归一化
    {"nativeId": "L6", "a_end_ne_nativeId": "", "z_end_ne_nativeId": "",
     "a_end_ne_nativeId(')": "ne3", "z_end_ne_nativeId(')": "ne4", "link_layer": "IP",
     "a_end_port_name": "pv1", "z_end_port_name": "pv2", "a_end_port_ip": "", "z_end_port_ip": "",
     "a_end_ne_manager_name": "", "z_end_ne_manager_name": "", "last_Modified": "1000"},
    # 无 key（无 nativeId/resId/source_uuid）-> 原样保留、不去重
    {"nativeId": "", "a_end_ne_nativeId": "ne5", "z_end_ne_nativeId": "ne6", "link_layer": "OTN",
     "a_end_port_name": "pn1", "z_end_port_name": "pn2", "a_end_port_ip": "", "z_end_port_ip": "",
     "a_end_ne_manager_name": "", "z_end_ne_manager_name": "", "last_Modified": "1000"},
]


def _write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _write_zip_csv(zip_path, member_name, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, buf.getvalue())


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_dataset(base: Path):
    ne_dir = base / "ne"
    site_dir = base / "site"
    link_dir = base / "links"
    for d in (ne_dir, site_dir, link_dir):
        d.mkdir(parents=True, exist_ok=True)

    # NE：拆成一个 csv 和一个 zip，验证目录合并 + zip 读取
    _write_csv(ne_dir / "SYS_NE_part1.csv", NE_ROWS[:3])
    _write_zip_csv(ne_dir / "SYS_NE_part2.zip", "SYS_NE_part2.csv", NE_ROWS[3:])

    # 站点：单 csv
    _write_csv(site_dir / "SYS_SITE_part.csv", SITE_ROWS)

    # 链路：jsonl + csv + zip(两份csv) 同放一个目录
    _write_jsonl(link_dir / "links_a.jsonl", LINKS_JSONL)
    _write_csv(link_dir / "links_b.csv", LINKS_CSV)
    with zipfile.ZipFile(link_dir / "links_c.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for member, rows in (("inner1.csv", LINKS_ZIP), ("inner2.csv", LINKS_ZIP_VARIANT)):
            buf = io.StringIO()
            fields = []
            for r in rows:
                for k in r:
                    if k not in fields:
                        fields.append(k)
            w = csv.DictWriter(buf, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
            zf.writestr(member, buf.getvalue())

    return ne_dir, site_dir, link_dir


# --------------------------------------------------------------------------- #
# 执行 & 比较
# --------------------------------------------------------------------------- #
def _run(args):
    res = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(
            f"命令失败: {' '.join(args)}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_equivalence_check():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ne_dir, site_dir, link_dir = build_dataset(base)

        orig = base / "orig"
        merged = base / "merged"
        orig.mkdir()
        merged.mkdir()

        # 分别执行三个独立工具
        _run(["topology_tools.extract_site_graph",
              "--ne-dir", str(ne_dir), "--site-dir", str(site_dir),
              "--link-input", str(link_dir), "-o", str(orig / "site_graph.json")])
        _run(["topology_tools.extract_ne_graph",
              "--ne-dir", str(ne_dir), "--site-graph", str(orig / "site_graph.json"),
              "--link-input", str(link_dir), "-o", str(orig / "ne_graph.json")])
        _run(["topology_tools.build_link_peer_index",
              "--sys-link", str(link_dir), "--output", str(orig / "link_peer_index.json")])

        # 一次性执行合并工具
        _run(["topology_tools.build_topology_graphs",
              "--ne-dir", str(ne_dir), "--site-dir", str(site_dir), "--link-input", str(link_dir),
              "--site-graph-output", str(merged / "site_graph.json"),
              "--ne-graph-output", str(merged / "ne_graph.json"),
              "--peer-index-output", str(merged / "link_peer_index.json")])

        failures = []
        for name in ("site_graph.json", "ne_graph.json", "link_peer_index.json"):
            a = _load(orig / name)
            b = _load(merged / name)
            if a != b:
                failures.append(name)
                print(f"[FAIL] {name} 内容不一致")
                only_orig = set(a) - set(b)
                only_merged = set(b) - set(a)
                if only_orig:
                    print(f"  仅原工具有的 key: {sorted(only_orig)}")
                if only_merged:
                    print(f"  仅合并工具有的 key: {sorted(only_merged)}")
                for key in set(a) & set(b):
                    if a[key] != b[key]:
                        print(f"  key={key} 值不同:\n    orig  ={a[key]}\n    merged={b[key]}")
            else:
                print(f"[OK]   {name} 内容一致（{len(a)} 个 key）")

        if failures:
            raise AssertionError(f"以下产物不一致: {failures}")
        print("\n全部一致 ✅")


def test_equivalence():
    run_equivalence_check()


if __name__ == "__main__":
    run_equivalence_check()
