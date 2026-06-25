import argparse

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from topology_resources import LINK_PEER_INDEX_JSON, SYS_LINK_JSONL, resource_display
from topology_tools.link_peer_index import build_peer_index_from_sys_link, save_peer_index


def main():
    parser = argparse.ArgumentParser(description="从 SYS_LINK 资源生成设备端口对端索引")
    parser.add_argument(
        "--output",
        default=LINK_PEER_INDEX_JSON,
        help=f"peer_index JSON 输出路径，默认: {resource_display('link_peer_index.json')}",
    )
    parser.add_argument(
        "--sys-link",
        default=SYS_LINK_JSONL,
        help=f"SYS_LINK 输入文件或目录，默认: {resource_display('sys_link_1231.jsonl')}",
    )
    parser.add_argument(
        "--report-duplicates",
        action="store_true",
        help="打印 SYS_LINK 链路 ID 重复记录统计",
    )
    args = parser.parse_args()

    peer_index = build_peer_index_from_sys_link(
        args.sys_link,
        report_duplicates=args.report_duplicates,
    )
    save_peer_index(peer_index, args.output)
    print(f"peer_index 已写出: {args.output}，记录数: {len(peer_index)}")


if __name__ == "__main__":
    main()
