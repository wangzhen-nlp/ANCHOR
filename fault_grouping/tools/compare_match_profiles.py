#!/usr/bin/env python3
"""Compare key matching call sites in two cProfile result files."""

import argparse
import pstats
from pathlib import Path


DEFAULT_PATTERN = (
    r"_evaluate_rule|validate_node|matches_node_structure|"
    r"_collect_pending_matches|_collect_mature_pending|"
    r"_collect_instance_edge_targets|_fork_primitive_target_instances|"
    r"_traverse_graph_role_filtered|write_matches|build_jsonl_match_output"
)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="对比普通版和 official 版的 cProfile 关键匹配调用。"
    )
    parser.add_argument(
        "--main-profile",
        default="/tmp/main.prof",
        help="普通版 cProfile 文件，默认: /tmp/main.prof",
    )
    parser.add_argument(
        "--official-profile",
        default="/tmp/official.prof",
        help="official 版 cProfile 文件，默认: /tmp/official.prof",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="传给 pstats.print_stats 的函数名正则表达式。",
    )
    parser.add_argument(
        "--sort",
        choices=("cumulative", "tottime", "calls"),
        default="cumulative",
        help="排序字段，默认: cumulative",
    )
    return parser


def _print_profile(name, profile_path, pattern, sort_key):
    print(f"\n========== {name} ({profile_path}) ==========")
    (
        pstats.Stats(str(profile_path))
        .strip_dirs()
        .sort_stats(sort_key)
        .print_stats(pattern)
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()
    profiles = (
        ("main", Path(args.main_profile).expanduser()),
        ("official", Path(args.official_profile).expanduser()),
    )

    missing = [str(path) for _, path in profiles if not path.is_file()]
    if missing:
        parser.error("找不到 profile 文件: " + ", ".join(missing))

    for name, profile_path in profiles:
        _print_profile(name, profile_path, args.pattern, args.sort)


if __name__ == "__main__":
    main()
