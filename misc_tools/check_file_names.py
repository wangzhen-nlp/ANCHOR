#!/usr/bin/env python3
"""Check whether filename stems from a list appear in one directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_existing_stems(directory: Path) -> set[str]:
    """Return filename stems for direct files in directory."""
    return {path.stem for path in directory.iterdir() if path.is_file()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "输入一个文件名列表文件和一个目录，按列表逐行输出对应文件是否存在。"
            "目录中的文件可以带后缀，比较时使用去掉最后一个后缀后的文件名。"
        )
    )
    parser.add_argument("name_list", type=Path, help="文件名列表文件，一行一个，不含路径和后缀")
    parser.add_argument("directory", type=Path, help="要检查的目录，只检查该目录直属文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.name_list.is_file():
        raise SystemExit(f"名单文件不存在或不是文件: {args.name_list}")
    if not args.directory.is_dir():
        raise SystemExit(f"目标目录不存在或不是目录: {args.directory}")

    existing_stems = build_existing_stems(args.directory)

    with args.name_list.open("r", encoding="utf-8") as file:
        for line in file:
            name = line.rstrip("\n\r").strip()
            print("是" if name and name in existing_stems else "否")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
