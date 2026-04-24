import json
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(2)

from topology_resources import NE_GRAPH_JSON, resource_display
from ticket_recall.evaluation.compute_group_output_ticket_recall import (
    compute_group_output_ticket_recall,
)
from ticket_recall.evaluation.compute_ticket_site_recall import (
    compute_ticket_site_recall,
)
from ticket_recall.ticket_recall_utils import (
    build_visualization_case_record,
    build_site_coord_index,
    build_site_coord_index_from_site_graph,
    build_site_to_ne_ids,
    derive_case_jsonl_output_path,
    load_ne_graph_data,
    load_site_graph_data,
    write_jsonl_records,
)


def _index_details_by_ticket_id(result):
    details = result.get("details", [])
    return {
        str(item.get("ticket_id", "")).strip(): item
        for item in details
        if isinstance(item, dict) and str(item.get("ticket_id", "")).strip()
    }


def _index_cases_by_ticket_id(case_records):
    return {
        str(item.get("ticket_id", "")).strip(): item
        for item in (case_records or [])
        if isinstance(item, dict) and str(item.get("ticket_id", "")).strip()
    }


def _format_sites(site_list):
    values = [str(site_id).strip() for site_id in (site_list or []) if str(site_id).strip()]
    return ", ".join(values) if values else "-"


def _build_method_case_payload(
    method_key,
    method_label,
    detail,
    original_case,
    ne_graph_data,
    site_to_ne_ids,
    site_coord_index,
):
    if not detail:
        return {
            "method_key": method_key,
            "method_label": method_label,
            "present": False,
        }

    viewer_case = original_case or build_visualization_case_record(
        detail,
        method=method_key,
        ne_graph_data=ne_graph_data,
        site_to_ne_ids=site_to_ne_ids,
        site_coord_index=site_coord_index,
    )
    associated_sites = detail.get("associated_sites", [])
    missing_sites = detail.get("missing_sites", [])
    recall = float(detail.get("recall", 0.0) or 0.0)
    precision = float(detail.get("precision", 0.0) or 0.0)
    f1 = float(detail.get("f1", 0.0) or 0.0)
    ticket_site_count = int(detail.get("ticket_site_count", len(detail.get("ticket_sites", []))) or 0)
    associated_site_count = int(detail.get("associated_site_count", len(associated_sites)) or 0)

    return {
        "method_key": method_key,
        "method_label": method_label,
        "present": True,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "ticket_site_count": ticket_site_count,
        "associated_site_count": associated_site_count,
        "missing_site_count": int(detail.get("missing_site_count", len(missing_sites)) or 0),
        "group_site_count": int(detail.get("group_site_count", 0) or 0),
        "fault_group_count": int(detail.get("fault_group_count", 0) or 0),
        "ticket_sites": detail.get("ticket_sites", []),
        "associated_sites": associated_sites,
        "missing_sites": missing_sites,
        "fault_groups": detail.get("fault_groups", []),
        "selected_fault_group": detail.get("selected_fault_group", ""),
        "detail": detail,
        "original_case": original_case,
        "viewer_case": viewer_case,
        "has_original_case": bool(original_case),
        "recall_text": f"{associated_site_count}/{ticket_site_count} = {recall:.6f}" if ticket_site_count else f"{recall:.6f}",
    }


def _build_combined_case_record(ticket_id, alarm_stream_payload, group_output_payload):
    preferred_payload = alarm_stream_payload if alarm_stream_payload.get("present") else group_output_payload
    ticket_sites = preferred_payload.get("ticket_sites", []) if preferred_payload else []
    ticket_site_count = int(preferred_payload.get("ticket_site_count", len(ticket_sites)) or 0) if preferred_payload else 0

    note_lines = [
        f"告警流召回站点列表：{_format_sites(alarm_stream_payload.get('associated_sites', []))}",
        f"故障组输出召回站点列表：{_format_sites(group_output_payload.get('associated_sites', []))}",
    ]

    return {
        "uuid": f"combined_ticket_recall::{ticket_id}",
        "ticket_id": ticket_id,
        "ticket_sites": ticket_sites,
        "ticket_site_count": ticket_site_count,
        "note": "\n".join(note_lines),
        "alarm_stream": alarm_stream_payload,
        "group_output": group_output_payload,
    }


def compute_ticket_recall_combined(
    alarms_input,
    group_output_input,
    upper_bound_file,
    ticket_sites_file=None,
    ticket_field="工单号",
    group_field="故障组ID",
    ne_graph_file=NE_GRAPH_JSON,
    output_file="ticket_recall_combined.json",
    combined_case_jsonl_output_file=None,
    only_offline=False,
    no_domain_alarm="",
    no_domain_site="",
    require_domain_per_site="",
    loose=False,
    potential=False,
    only_one=False,
    ultimate_only=False,
    min_site_num=0,
    upper_bound_associated_as_gold=False,
):
    if combined_case_jsonl_output_file is None:
        combined_case_jsonl_output_file = derive_case_jsonl_output_path(output_file)

    print("步骤 1/3：计算告警流评测结果...")
    alarm_stream_result = compute_ticket_site_recall(
        alarm_input=alarms_input,
        upper_bound_file=upper_bound_file,
        ticket_sites_file=ticket_sites_file,
        ticket_field=ticket_field,
        group_field=group_field,
        ne_graph_file=ne_graph_file,
        output_file=None,
        case_jsonl_output_file=None,
        only_offline=only_offline,
        no_domain_alarm=no_domain_alarm,
        no_domain_site=no_domain_site,
        require_domain_per_site=require_domain_per_site,
        loose=loose,
        potential=potential,
        only_one=only_one,
        min_site_num=min_site_num,
        upper_bound_associated_as_gold=upper_bound_associated_as_gold,
    )

    print("步骤 2/3：计算故障组输出评测结果...")
    group_output_result = compute_group_output_ticket_recall(
        group_output_input=group_output_input,
        upper_bound_file=upper_bound_file,
        ticket_sites_file=ticket_sites_file,
        ticket_field=ticket_field,
        alarms_input=alarms_input,
        ne_graph_file=ne_graph_file,
        output_file=None,
        case_jsonl_output_file=None,
        only_offline=only_offline,
        no_domain_alarm=no_domain_alarm,
        no_domain_site=no_domain_site,
        require_domain_per_site=require_domain_per_site,
        loose=loose,
        potential=potential,
        only_one=only_one,
        ultimate_only=ultimate_only,
        min_site_num=min_site_num,
        upper_bound_associated_as_gold=upper_bound_associated_as_gold,
    )

    print("步骤 3/3：整合两个结果并生成 combined cases...")
    ne_graph_data = load_ne_graph_data(ne_graph_file)
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    site_coord_index = build_site_coord_index(ne_graph_data)
    site_coord_index.update(
        build_site_coord_index_from_site_graph(load_site_graph_data())
    )

    alarm_stream_details = _index_details_by_ticket_id(alarm_stream_result)
    group_output_details = _index_details_by_ticket_id(group_output_result)

    all_ticket_ids = sorted(set(alarm_stream_details) | set(group_output_details))
    combined_case_records = []
    for ticket_id in all_ticket_ids:
        alarm_stream_payload = _build_method_case_payload(
            method_key="alarm_stream_group_field",
            method_label="告警流",
            detail=alarm_stream_details.get(ticket_id),
            original_case=None,
            ne_graph_data=ne_graph_data,
            site_to_ne_ids=site_to_ne_ids,
            site_coord_index=site_coord_index,
        )
        group_output_payload = _build_method_case_payload(
            method_key="group_output",
            method_label="故障组输出",
            detail=group_output_details.get(ticket_id),
            original_case=None,
            ne_graph_data=ne_graph_data,
            site_to_ne_ids=site_to_ne_ids,
            site_coord_index=site_coord_index,
        )
        combined_case_records.append(
            _build_combined_case_record(ticket_id, alarm_stream_payload, group_output_payload)
        )

    combined_case_records.sort(
        key=lambda record: (
            float(((record.get("group_output") or {}).get("recall", 0.0) or 0.0)),
            str(record.get("ticket_id", "")),
        )
    )

    write_jsonl_records(combined_case_jsonl_output_file, combined_case_records)

    combined_result = {
        "method": "ticket_recall_combined",
        "combined_case_jsonl_output": combined_case_jsonl_output_file,
        "ticket_count": len(combined_case_records),
        "alarm_stream_summary": {
            "ticket_count": alarm_stream_result.get("ticket_count", 0),
            "final_sample_count": alarm_stream_result.get("final_sample_count", 0),
            "ticket_site_count_distribution": alarm_stream_result.get("ticket_site_count_distribution", {}),
            "average_recall": alarm_stream_result.get("average_recall", 0.0),
            "average_precision": alarm_stream_result.get("average_precision", 0.0),
            "average_f1": alarm_stream_result.get("average_f1", 0.0),
        },
        "group_output_summary": {
            "ticket_count": group_output_result.get("ticket_count", 0),
            "final_sample_count": group_output_result.get("final_sample_count", 0),
            "ticket_site_count_distribution": group_output_result.get("ticket_site_count_distribution", {}),
            "average_recall": group_output_result.get("average_recall", 0.0),
            "average_precision": group_output_result.get("average_precision", 0.0),
            "average_f1": group_output_result.get("average_f1", 0.0),
            "denominator_source": group_output_result.get("denominator_source", ""),
        },
        "shared_options": {
            "only_offline_mode": only_offline,
            "no_domain_alarm": no_domain_alarm,
            "no_domain_alarm_mode": bool(no_domain_alarm),
            "no_domain_site": no_domain_site,
            "no_domain_site_mode": bool(no_domain_site),
            "require_domain_per_site": require_domain_per_site,
            "require_domain_per_site_mode": bool(require_domain_per_site),
            "loose_mode": loose,
            "potential_mode": potential,
            "only_one_mode": only_one,
            "ultimate_only_mode": ultimate_only,
            "min_site_num": min_site_num,
            "upper_bound_associated_as_gold_mode": upper_bound_associated_as_gold,
        },
        "details": [
            {
                "ticket_id": record["ticket_id"],
                "ticket_site_count": record["ticket_site_count"],
                "ticket_sites": record["ticket_sites"],
                "alarm_stream": {
                    "present": record["alarm_stream"].get("present", False),
                    "recall": record["alarm_stream"].get("recall", 0.0),
                    "precision": record["alarm_stream"].get("precision", 0.0),
                    "f1": record["alarm_stream"].get("f1", 0.0),
                    "associated_site_count": record["alarm_stream"].get("associated_site_count", 0),
                    "associated_sites": record["alarm_stream"].get("associated_sites", []),
                    "missing_site_count": record["alarm_stream"].get("missing_site_count", 0),
                    "missing_sites": record["alarm_stream"].get("missing_sites", []),
                    "fault_group_count": record["alarm_stream"].get("fault_group_count", 0),
                    "fault_groups": record["alarm_stream"].get("fault_groups", []),
                },
                "group_output": {
                    "present": record["group_output"].get("present", False),
                    "recall": record["group_output"].get("recall", 0.0),
                    "precision": record["group_output"].get("precision", 0.0),
                    "f1": record["group_output"].get("f1", 0.0),
                    "associated_site_count": record["group_output"].get("associated_site_count", 0),
                    "associated_sites": record["group_output"].get("associated_sites", []),
                    "missing_site_count": record["group_output"].get("missing_site_count", 0),
                    "missing_sites": record["group_output"].get("missing_sites", []),
                    "fault_group_count": record["group_output"].get("fault_group_count", 0),
                    "fault_groups": record["group_output"].get("fault_groups", []),
                },
            }
            for record in combined_case_records
        ],
    }
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, ensure_ascii=False, indent=2)
    return combined_result


def main():
    parser = ArgumentParser(
        description="整合两个评测脚本，分别输出原始两份结果，并额外生成一个按工单聚合的 combined cases.jsonl"
    )
    parser.add_argument("alarms", help="原始告警输入，支持 jsonl/csv/zip/目录")
    parser.add_argument("group_output", help="match_rules.py 的 group 输出，支持 jsonl/zip/目录")
    parser.add_argument(
        "--upper-bound",
        required=True,
        help="compute_ticket_site_recall_upper_bound.py 的输出 JSON",
    )
    parser.add_argument(
        "--ticket-sites",
        help="工单站点映射 JSON；不提供时会退化为从 alarms 中回推工单站点",
    )
    parser.add_argument(
        "--ticket-field",
        default="工单号",
        help="工单字段名，默认: 工单号",
    )
    parser.add_argument(
        "--group-field",
        default="故障组ID",
        help="告警流中的故障组字段名，默认: 故障组ID",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"用于回填站点和构造可视化的 ne_graph 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ticket_recall_combined.json",
        help="整合后的汇总 JSON 输出，默认: ticket_recall_combined.json",
    )
    parser.add_argument(
        "--combined-case-jsonl-output",
        help="整合后的工单级 cases 输出；默认随主输出生成同名 .cases.jsonl",
    )
    parser.add_argument(
        "--only-offline",
        action="store_true",
        help="仅保留 upper bound evidence 中出现过 OFFLINE_ALARMS 的工单样本",
    )
    parser.add_argument(
        "--no-domain-alarm",
        metavar="DOMAIN",
        help="如果 upper bound evidence 中存在来自指定 domain 的告警，则跳过该工单样本，例如: --no-domain-alarm DATA",
    )
    parser.add_argument(
        "--no-domain-site",
        metavar="DOMAIN",
        help="如果当前工单站点里存在包含指定 domain 设备的站点，则跳过该工单样本，例如: --no-domain-site DATA",
    )
    parser.add_argument(
        "--require-domain-per-site",
        metavar="DOMAIN",
        help="先从工单站点里剔除不包含指定 domain 设备的站点；过滤后若站点数不足 min-site-num，则跳过该工单，例如: --require-domain-per-site TRANSMISSION",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="允许用 upper bound 同口径时间窗扩充额外 group / 故障组ID",
    )
    parser.add_argument(
        "--potential",
        action="store_true",
        help="允许用 upper bound evidence 中出现过的告警，直接吸附这些告警所在的额外 group / 故障组ID",
    )
    parser.add_argument(
        "--only-one",
        action="store_true",
        help="只保留覆盖该工单目标站点最多的单个 group / 故障组ID，用它的站点计算召回率",
    )
    parser.add_argument(
        "--ultimate-only",
        action="store_true",
        help="只用于故障组输出：只考虑不作为关联 group 出现的最终 group",
    )
    parser.add_argument(
        "--min-site-num",
        type=int,
        default=0,
        help="仅统计工单站点数 >= 该值的工单；默认: 0",
    )
    parser.add_argument(
        "--upper-bound-associated-as-gold",
        action="store_true",
        help="改用 upper bound 的 associated_sites 作为 gold",
    )

    args = parser.parse_args()

    try:
        result = compute_ticket_recall_combined(
            alarms_input=args.alarms,
            group_output_input=args.group_output,
            upper_bound_file=args.upper_bound,
            ticket_sites_file=args.ticket_sites,
            ticket_field=args.ticket_field,
            group_field=args.group_field,
            ne_graph_file=args.ne_graph,
            output_file=args.output,
            combined_case_jsonl_output_file=args.combined_case_jsonl_output,
            only_offline=args.only_offline,
            no_domain_alarm=args.no_domain_alarm,
            no_domain_site=args.no_domain_site,
            require_domain_per_site=args.require_domain_per_site,
            loose=args.loose,
            potential=args.potential,
            only_one=args.only_one,
            ultimate_only=args.ultimate_only,
            min_site_num=args.min_site_num,
            upper_bound_associated_as_gold=args.upper_bound_associated_as_gold,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    print(f"整合工单数: {result['ticket_count']}")
    alarm_stream_site_distribution = result["alarm_stream_summary"].get("ticket_site_count_distribution", {})
    group_output_site_distribution = result["group_output_summary"].get("ticket_site_count_distribution", {})
    if alarm_stream_site_distribution == group_output_site_distribution:
        print(f"保留工单召回站点数分布: {alarm_stream_site_distribution}")
    else:
        print(f"告警流保留工单召回站点数分布: {alarm_stream_site_distribution}")
        print(f"故障组输出保留工单召回站点数分布: {group_output_site_distribution}")
    print(
        "告警流指标: "
        f"平均召回率={result['alarm_stream_summary']['average_recall']:.6f}, "
        f"平均准确率={result['alarm_stream_summary']['average_precision']:.6f}, "
        f"平均F1={result['alarm_stream_summary']['average_f1']:.6f}"
    )
    print(
        "故障组输出指标: "
        f"平均召回率={result['group_output_summary']['average_recall']:.6f}, "
        f"平均准确率={result['group_output_summary']['average_precision']:.6f}, "
        f"平均F1={result['group_output_summary']['average_f1']:.6f}"
    )
    print(f"整合汇总 JSON: {args.output}")
    print(f"combined cases: {result['combined_case_jsonl_output']}")


if __name__ == "__main__":
    main()
