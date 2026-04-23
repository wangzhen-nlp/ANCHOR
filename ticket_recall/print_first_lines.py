import argparse
import gzip
import io
import os
import sys
import zipfile
from contextlib import contextmanager


@contextmanager
def _open_input(path: str, encoding: str, zip_member: str = ""):
    if path == "-":
        yield sys.stdin
        return

    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding, errors="replace") as f:
            yield f
        return

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")]
            if not members:
                raise ValueError(f"zip 文件中没有可读取的文件: {path}")
            selected_member = zip_member or members[0]
            if selected_member not in members:
                raise ValueError(f"zip 文件中不存在指定成员: {selected_member}")
            with zf.open(selected_member, "r") as raw:
                with io.TextIOWrapper(raw, encoding=encoding, errors="replace") as f:
                    yield f
        return

    with open(path, "r", encoding=encoding, errors="replace") as f:
        yield f


@contextmanager
def _open_output(path: str, encoding: str):
    if not path or path == "-":
        yield sys.stdout
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        yield f


def write_first_lines(input_file: str, output_file: str = "", num_lines: int = 10, encoding: str = "utf-8", zip_member: str = "") -> int:
    if num_lines < 0:
        raise ValueError("num_lines 不能为负数")

    written = 0
    with _open_input(input_file, encoding, zip_member=zip_member) as fin:
        with _open_output(output_file, encoding) as fout:
            for line in fin:
                if written >= num_lines:
                    break
                fout.write(line)
                written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="输出输入文件的前 n 行")
    parser.add_argument("input", help="输入文件；传 '-' 表示从 stdin 读取")
    parser.add_argument(
        "-n",
        "--num-lines",
        type=int,
        default=10,
        help="输出前多少行，默认: 10",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="输出文件；不提供或传 '-' 表示输出到 stdout",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码，默认: utf-8",
    )
    parser.add_argument(
        "--zip-member",
        default="",
        help="输入为 zip 时读取指定成员；默认读取 zip 内第一个文件",
    )

    args = parser.parse_args()
    written = write_first_lines(
        input_file=args.input,
        output_file=args.output,
        num_lines=args.num_lines,
        encoding=args.encoding,
        zip_member=args.zip_member,
    )
    if args.output and args.output != "-":
        print(f"已输出前 {written} 行到: {args.output}")


if __name__ == "__main__":
    main()
