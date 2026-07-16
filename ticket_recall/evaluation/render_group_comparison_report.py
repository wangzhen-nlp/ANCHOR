#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 compute_ultimate_group_alarm_group_metrics.py 的指标 JSON 渲染成自包含 HTML 对比报告。

报告并排展示两个方向（终极 group 作为 gold / 告警故障组ID 作为 gold）的站点级与告警级
召回率、准确率、F1，附 gold 站点数分布和最差样本明细。

用法：
    python ticket_recall/evaluation/render_group_comparison_report.py metrics.json -o report.html
"""

import html
import json

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(2)

# compare_visual_alarm_groups.py 用 mhp_group_as_gold，compute_ultimate_group_alarm_group_metrics.py
# 用 ultimate_group_as_gold；两者结构一致，按实际存在的键渲染。
DIRECTION_CANDIDATES = (
    ("mhp_group_as_gold", "MHP 生成的故障组 作为 gold", "MHP 生成的故障组当基准，看告警故障组ID能否复现它"),
    ("ultimate_group_as_gold", "终极 group 作为 gold", "生成的故障组当基准，看告警故障组ID能否复现它"),
    ("alarm_group_as_gold", "告警故障组ID 作为 gold", "告警故障组ID当基准，看生成的故障组能否复现它"),
)


def _resolve_directions(metrics):
    return [
        (key, title, hint)
        for key, title, hint in DIRECTION_CANDIDATES
        if isinstance(metrics.get(key), dict)
    ]

METRIC_ROWS = (
    ("召回率", "average_recall", "average_alarm_recall"),
    ("准确率", "average_precision", "average_alarm_precision"),
    ("F1", "average_f1", "average_alarm_f1"),
)

STYLE = """
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: "Segoe UI", "PingFang SC", Tahoma, sans-serif;
    background: #f4f7fb;
    color: #1f2937;
}
.page { max-width: 1360px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 12px; }
.subtitle { color: #6b7280; font-size: 13px; margin-bottom: 20px; }
.card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 18px;
}
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #e5e7eb; padding: 7px 9px; text-align: left; }
th { background: #f9fafb; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.metric-table td.num { font-size: 15px; font-weight: 600; }
.flags { font-size: 12px; color: #4b5563; }
.flags code {
    background: #f3f4f6;
    border-radius: 4px;
    padding: 1px 5px;
    margin-right: 6px;
}
.warn {
    background: #fef2f2;
    border-color: #fecaca;
    color: #991b1b;
}
.note {
    background: #fffbeb;
    border-color: #fde68a;
    color: #92400e;
}
.bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; font-size: 12px; }
.bar-label { width: 88px; color: #6b7280; }
.bar-track { flex: 1; background: #f3f4f6; border-radius: 3px; height: 14px; }
.bar-fill { background: #60a5fa; height: 14px; border-radius: 3px; }
.bar-value { width: 64px; text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
details { margin-top: 10px; }
summary { cursor: pointer; font-size: 13px; color: #2563eb; }
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: 1px 8px;
    font-size: 11px;
    background: #eef2ff;
    color: #3730a3;
}
"""


def _esc(value):
    return html.escape(str(value))


def _fmt_ratio(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _render_flags(metrics):
    flags = [
        ("group_field", metrics.get("group_field", "")),
        ("min_site_num", metrics.get("min_site_num", 0)),
    ]
    for name in (
        "no_domain_alarm",
        "no_domain_site",
        "require_domain_per_site",
    ):
        if metrics.get(name):
            flags.append((name, metrics[name]))
    for name in ("only_offline_mode", "only_one_mode", "loose_mode", "potential_mode"):
        if metrics.get(name):
            flags.append((name.replace("_mode", ""), "on"))
    if metrics.get("loose_mode"):
        flags.append(("window_seconds", metrics.get("window_seconds", 0)))

    parts = "".join(f"<code>{_esc(k)}={_esc(v)}</code>" for k, v in flags)
    return f'<div class="flags">{parts}</div>'


def _render_symptom_scope(metrics):
    if not metrics.get("virtual_symptoms_excluded"):
        return ""
    return f"""
<div class="card note">
  <h2>告警口径</h2>
  <p>生成组一侧：真实 symptom {_fmt_int(metrics.get('real_symptom_count'))} 条，
  其中带故障组ID的 {_fmt_int(metrics.get('symptom_with_group_id_count'))} 条；
  已剔除虚拟/推断 symptom {_fmt_int(metrics.get('skipped_virtual_symptom_count'))} 条
  （模型补出来的告警，原始流里不存在，occurrence_uuid 是随机生成的，不参与指标）。</p>
</div>
"""


def _render_overlap(overlap):
    shared = int(overlap.get("shared_alarm_count", 0) or 0)
    card_class = "card warn" if shared == 0 else "card"
    warning = ""
    if shared == 0:
        warning = (
            "<p><strong>两侧没有任何共有告警实例，告警级指标不可信。</strong>"
            "occurrence_uuid 由原始告警记录内容取哈希，请确认 group output 与 alarms 来自同一份告警导出。</p>"
        )
    return f"""
<div class="{card_class}">
  <h2>告警实例键重合度</h2>
  {warning}
  <table>
    <tr><th>生成组侧告警数</th><td class="num">{_fmt_int(overlap.get('ultimate_side_alarm_count'))}</td>
        <th>其中两侧共有占比</th><td class="num">{_fmt_ratio(overlap.get('ultimate_side_shared_ratio'))}</td></tr>
    <tr><th>告警故障组ID 侧告警数</th><td class="num">{_fmt_int(overlap.get('alarm_group_side_alarm_count'))}</td>
        <th>其中两侧共有占比</th><td class="num">{_fmt_ratio(overlap.get('alarm_group_side_shared_ratio'))}</td></tr>
    <tr><th>两侧共有告警数</th><td class="num">{_fmt_int(overlap.get('shared_alarm_count'))}</td>
        <th>Jaccard</th><td class="num">{_fmt_ratio(overlap.get('jaccard'))}</td></tr>
  </table>
</div>
"""


def _render_distribution(distribution):
    if not distribution:
        return "<p class='flags'>无分布数据</p>"
    items = sorted(distribution.items(), key=lambda kv: int(kv[0]))
    peak = max(int(count) for _label, count in items) or 1
    rows = []
    for label, count in items:
        width = int(count) / peak * 100
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{_esc(label)} 个站点</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width:.2f}%"></span></span>'
            f'<span class="bar-value">{_fmt_int(count)}</span></div>'
        )
    return "".join(rows)


def _render_worst_samples(details, limit):
    if not details:
        return "<p class='flags'>没有明细数据</p>"

    ranked = sorted(
        details,
        key=lambda item: (float(item.get("f1", 0.0) or 0.0), -int(item.get("gold_site_count", 0) or 0)),
    )[:limit]

    rows = []
    for item in ranked:
        missing_sites = sorted(set(item.get("gold_sites", [])) - set(item.get("matched_sites", [])))
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('gold_id', ''))}</td>"
            f"<td class='num'>{_fmt_int(item.get('gold_site_count'))}</td>"
            f"<td class='num'>{_fmt_int(item.get('effective_predicted_group_count'))}</td>"
            f"<td class='num'>{_fmt_ratio(item.get('recall'))}</td>"
            f"<td class='num'>{_fmt_ratio(item.get('precision'))}</td>"
            f"<td class='num'>{_fmt_ratio(item.get('f1'))}</td>"
            f"<td class='num'>{_fmt_ratio(item.get('alarm_recall'))}</td>"
            f"<td class='num'>{_fmt_ratio(item.get('alarm_precision'))}</td>"
            f"<td class='num'>{_fmt_ratio(item.get('alarm_f1'))}</td>"
            f"<td class='num'>{_fmt_int(item.get('gold_alarms_missing_from_pred_universe_count'))}"
            f"/{_fmt_int(item.get('gold_alarm_count'))}</td>"
            f"<td>{_esc('，'.join(missing_sites) or '无')}</td>"
            "</tr>"
        )

    return f"""
<div class="scroll">
<table>
  <tr>
    <th>gold 组 ID</th><th class="num">gold站点数</th><th class="num">命中预测组数</th>
    <th class="num">站点召回</th><th class="num">站点准确</th><th class="num">站点F1</th>
    <th class="num">告警召回</th><th class="num">告警准确</th><th class="num">告警F1</th>
    <th class="num">对侧缺失告警</th><th>未召回站点</th>
  </tr>
  {''.join(rows)}
</table>
</div>
"""


def _render_direction(metrics, key, title, hint, worst_limit):
    section = metrics.get(key, {})
    details = section.get("details", [])

    metric_rows = "".join(
        f"<tr><th>{_esc(label)}</th>"
        f"<td class='num'>{_fmt_ratio(section.get(site_key))}</td>"
        f"<td class='num'>{_fmt_ratio(section.get(alarm_key))}</td></tr>"
        for label, site_key, alarm_key in METRIC_ROWS
    )

    missing_total = section.get("gold_alarms_missing_from_pred_universe_total", 0)
    gold_alarm_total = section.get("gold_alarm_total", 0)
    missing_ratio = (missing_total / gold_alarm_total) if gold_alarm_total else 0.0
    missing_note = ""
    if missing_total:
        missing_note = (
            f"<div class='card note' style='margin:12px 0 0;padding:10px'>"
            f"gold 告警中有 <strong>{_fmt_int(missing_total)}</strong> / {_fmt_int(gold_alarm_total)}"
            f"（{missing_ratio:.2%}）在对侧告警全集里根本不存在 —— 这部分不是分组分歧，"
            f"而是对侧压根没见过这条告警（被过滤、未进流，或没有故障组ID）。</div>"
        )

    details_filter_note = ""
    if section.get("details_filter"):
        details_filter_note = (
            f"<p class='flags'>明细已按 <code>{_esc(section['details_filter'])}</code> 过滤："
            f"输出 {_fmt_int(section.get('details_output_count'))} / "
            f"{_fmt_int(section.get('details_total_count'))} 条；平均指标仍基于全部样本。</p>"
        )

    return f"""
<div class="card">
  <h2>{_esc(title)} <span class="pill">样本数 {_fmt_int(section.get('sample_count'))}</span></h2>
  <p class="subtitle" style="margin-bottom:12px">{_esc(hint)}</p>
  <table class="metric-table">
    <tr><th style="width:120px">平均指标</th><th class="num">站点级</th><th class="num">告警级</th></tr>
    {metric_rows}
  </table>
  {missing_note}
  <h2 style="margin:18px 0 8px;font-size:14px">gold 站点数分布</h2>
  {_render_distribution(section.get('gold_site_count_distribution', {}))}
  <details open>
    <summary>最差 {worst_limit} 个样本（按站点级 F1 升序）</summary>
    {details_filter_note}
    {_render_worst_samples(details, worst_limit)}
  </details>
</div>
"""


def _render_scope_sections(metrics, worst_limit):
    """alarm-scope 为 both 时按口径分段渲染；否则退回顶层镜像的主口径。"""
    scopes = metrics.get("scopes")
    if not isinstance(scopes, dict) or len(scopes) <= 1:
        return f'<div class="grid">{_render_directions(metrics, worst_limit)}</div>'

    sections = []
    for scope_name, scope_result in scopes.items():
        overlap_html = ""
        if scope_name == "raw" and scope_result.get("alarm_identity_overlap"):
            overlap_html = _render_overlap(scope_result["alarm_identity_overlap"])
        sections.append(f"""
<h2 style="margin:26px 0 10px;font-size:19px">告警范围口径：{_esc(scope_result.get('alarm_scope_label', scope_name))}</h2>
<p class="subtitle">告警故障组 {_fmt_int(scope_result.get('alarm_group_count'))} 个</p>
{overlap_html}
<div class="grid">{_render_directions(scope_result, worst_limit)}</div>
""")
    return "".join(sections)


def _render_directions(section_metrics, worst_limit):
    return "".join(
        _render_direction(section_metrics, key, title, hint, worst_limit)
        for key, title, hint in _resolve_directions(section_metrics)
    )


def render_report(metrics, worst_limit=20):
    direction_cards = _render_scope_sections(metrics, worst_limit)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>故障组对比报告</title>
<style>{STYLE}</style>
</head>
<body>
<div class="page">
  <h1>生成故障组 vs 告警故障组ID 对比报告</h1>
  <p class="subtitle">
    生成的故障组 {_fmt_int(metrics.get('mhp_group_count', metrics.get('ultimate_group_count')))} 个，
    告警故障组ID 聚出 {_fmt_int(metrics.get('alarm_group_count'))} 个。
    两个方向互为基准，站点级与告警级两种粒度并列。
  </p>
  {_render_symptom_scope(metrics)}
  <div class="card">
    <h2>口径</h2>
    {_render_flags(metrics)}
  </div>
  {_render_overlap(metrics['alarm_identity_overlap']) if (metrics.get('alarm_identity_overlap') and not (isinstance(metrics.get('scopes'), dict) and len(metrics['scopes']) > 1)) else ''}
  {direction_cards}
</div>
</body>
</html>
"""


def main():
    parser = ArgumentParser(
        description="把 compute_ultimate_group_alarm_group_metrics.py 的指标 JSON 渲染成自包含 HTML 对比报告"
    )
    parser.add_argument(
        "metrics_json",
        help="compute_ultimate_group_alarm_group_metrics.py 的输出 JSON",
    )
    parser.add_argument(
        "--worst-limit",
        type=int,
        default=20,
        help="每个方向展示的最差样本条数，默认: 20",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="group_comparison_report.html",
        help="输出 HTML 文件，默认: group_comparison_report.html",
    )

    args = parser.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    document = render_report(metrics, worst_limit=args.worst_limit)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(document)

    print(f"报告已输出到: {args.output}")


if __name__ == "__main__":
    main()
