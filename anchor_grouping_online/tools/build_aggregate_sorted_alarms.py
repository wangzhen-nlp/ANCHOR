"""从 build_batch_stream_visualization.py 的输出反查指定二次汇聚组，
生成可直接喂给 match_rules.py 的排序告警缓存。

背景：
    build_batch_stream_visualization.py 会把滑窗二次汇聚输出整理成
    visualization JSONL，每行是一个二次汇聚组，``uuid`` 为二次汇聚 ID，
    ``symptoms`` 为该组去重后的成员告警。

本脚本：
    1. 读取上述 visualization JSONL；
    2. 按 ``--uuid`` 指定的一个或多个二次汇聚组 ID 过滤；
    3. 把命中组内的 symptoms 还原为 prepare_sorted_alarms.py 同款的
       告警事件（同一 alarm_events.io 结构），按 ts 排序；
    4. 用 alarm_events.sorted_cache.write_sorted_alarm_cache 写出排序告警缓存
       (JSONL/ZIP)，cache_type 与 fault_grouping / fault_grouping_official
       完全一致。

于是可直接：
    python fault_grouping/match_rules.py <cache> out.jsonl \
        --sorted-alarms-input <cache>
或让 match_rules 自动识别缓存格式：
    python fault_grouping_official/match_rules.py <cache> out.jsonl

用法：
    python anchor_grouping_online/tools/build_aggregate_sorted_alarms.py \
        secondary_aggregates_visualization.jsonl agg_sorted_alarms.jsonl \
        --uuid <二次汇聚组UUID> [--uuid <另一个UUID> ...]
"""

import json
import os
import time

from argparse import ArgumentParser
from datetime import datetime

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.alarm_events.sorted_cache import write_sorted_alarm_cache
from anchor_grouping_online.alarm_events.generator import port_vid_from_extendedattr


SITE_PLACEHOLDER_PREFIX = "SITE::"


def _format_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_alarm_source(raw_source):
    """去掉 build_batch_stream_visualization 为空告警源补的站点占位。"""
    source = str(raw_source or "").strip()
    if source.startswith(SITE_PLACEHOLDER_PREFIX):
        return ""
    return source


def _physical_port_name_from_extendedattr(extendedattr, alarm_source, line_number):
    port_vid = port_vid_from_extendedattr(extendedattr)
    if not port_vid:
        return ""
    if not alarm_source:
        raise ValueError(
            f"第 {line_number} 行 symptom 有 portVid 但缺少 alarm_source，"
            "无法还原物理端口名称"
        )
    expected_prefix = f"{alarm_source.upper()}|"
    if not port_vid.startswith(expected_prefix):
        raise ValueError(
            f"第 {line_number} 行 symptom 的 portVid 与 alarm_source 不匹配: "
            f"{port_vid!r} / {alarm_source!r}"
        )
    physical_port_name = port_vid[len(expected_prefix):]
    if not physical_port_name:
        raise ValueError(f"第 {line_number} 行 symptom 的 portVid 缺少端口名")
    return physical_port_name


def _symptom_to_alarm_event(symptom, line_number):
    if not isinstance(symptom, dict):
        raise ValueError(f"第 {line_number} 行 symptom 必须是对象")

    event_id = str(symptom.get("event_id", "") or "").strip()
    occurrence_uuid = str(symptom.get("occurrence_uuid", "") or "").strip()
    if not event_id:
        raise ValueError(f"第 {line_number} 行 symptom 缺少 event_id")
    if not occurrence_uuid:
        raise ValueError(f"第 {line_number} 行 symptom 缺少 occurrence_uuid")

    ts = symptom.get("ts")
    if not isinstance(ts, (int, float)):
        raise ValueError(f"第 {line_number} 行 symptom 缺少数值型 ts")
    ts = float(ts)

    site_id = str(symptom.get("node", "") or "").strip()
    alarm_title = str(symptom.get("alarm", "") or "").strip()
    alarm_source = _normalize_alarm_source(symptom.get("alarm_source"))
    physical_port_name = _physical_port_name_from_extendedattr(
        symptom.get("extendedattr", ""),
        alarm_source,
        line_number,
    )

    alarm = {
        "站点ID": site_id,
        "告警标题": alarm_title,
        "告警首次发生时间": _format_ts(ts),
        "告警编码ID": event_id,
        "告警源": alarm_source,
        "物理端口名称": physical_port_name,
    }
    return {
        "alarm": alarm,
        "site_id": site_id,
        "alarm_source": alarm_source,
        "alarm_title": alarm_title,
        "ts": ts,
        "occurrence_uuid": occurrence_uuid,
    }


def collect_aggregate_alarms(input_path, wanted_uuids):
    """流式读取 visualization JSONL，返回命中组去重后的告警事件列表。"""
    wanted = set(wanted_uuids)
    matched_uuids = set()
    events_by_identity = {}
    record_count = 0

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            line = line.strip()
            if not line:
                continue
            record_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行 JSON 解析失败: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行记录必须是对象")

            agg_id = str(record.get("uuid", "") or "").strip()
            if agg_id not in wanted:
                continue
            matched_uuids.add(agg_id)

            symptoms = record.get("symptoms", [])
            if not isinstance(symptoms, list):
                raise ValueError(f"第 {line_number} 行 symptoms 必须是列表")
            for symptom in symptoms:
                event = _symptom_to_alarm_event(symptom, line_number)
                identity = (event["alarm"]["告警编码ID"], event["occurrence_uuid"])
                # 同一二次汇聚组内已去重；跨组合并时按告警身份再去重一次。
                events_by_identity.setdefault(identity, event)

    missing = sorted(wanted - matched_uuids)
    if missing:
        raise ValueError(
            "以下二次汇聚组 UUID 在输入中未找到: " + ", ".join(missing)
        )

    events = list(events_by_identity.values())
    events.sort(key=lambda item: (item["ts"], item["alarm"]["告警编码ID"]))
    return events, record_count


def build_aggregate_sorted_alarms(input_path, output_path, wanted_uuids):
    events, record_count = collect_aggregate_alarms(input_path, wanted_uuids)

    metadata = {
        "source_visualization": os.path.abspath(input_path),
        "aggregate_uuids": sorted(set(wanted_uuids)),
        "clear_delay_sec": 0.0,
        "processed_count": len(events),
        "normal_alarm_count": len(events),
        "clear_alarm_count": 0,
        "cached_normal_alarm_count": len(events),
        "cached_clear_alarm_count": 0,
    }
    header = write_sorted_alarm_cache(output_path, events, metadata)
    return header, record_count


def _parse_uuids(raw_values):
    uuids = []
    seen = set()
    for raw_value in raw_values or []:
        for piece in str(raw_value).split(","):
            piece = piece.strip()
            if piece and piece not in seen:
                seen.add(piece)
                uuids.append(piece)
    return uuids


def main():
    parser = ArgumentParser(
        description="从二次汇聚可视化输出反查指定组，生成 match_rules 可用的排序告警缓存"
    )
    parser.add_argument(
        "input", help="build_batch_stream_visualization.py 输出的 visualization JSONL"
    )
    parser.add_argument(
        "output", help="排序告警缓存输出；后缀为 .zip 时写压缩包，否则写 JSONL"
    )
    parser.add_argument(
        "--uuid",
        action="append",
        required=True,
        help="指定的二次汇聚组 UUID；可重复传入，也支持逗号分隔",
    )
    args = parser.parse_args()

    wanted_uuids = _parse_uuids(args.uuid)
    if not wanted_uuids:
        parser.error("--uuid 不能为空")

    start_time = time.time()
    header, record_count = build_aggregate_sorted_alarms(
        args.input, args.output, wanted_uuids
    )
    elapsed = time.time() - start_time
    print(
        f"✅ 排序告警缓存已写入: {args.output}\n"
        f"   命中二次汇聚组: {len(wanted_uuids)} 个 ({', '.join(wanted_uuids)})\n"
        f"   扫描可视化记录: {record_count} 条\n"
        f"   缓存告警数: {header['alarm_count']}\n"
        f"   耗时: {elapsed:.2f} 秒"
    )


if __name__ == "__main__":
    main()
