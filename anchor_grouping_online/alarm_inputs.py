"""独立读取 JSONL、CSV 或 ZIP 告警输入。"""

import csv
import io
import json
import os
import zipfile

from anchor_grouping_online.tools.progress_utils import ProgressBar


def _iter_alarm_filepaths(path):
    if os.path.isdir(path):
        for filename in sorted(os.listdir(path)):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                yield filepath
        return
    yield path


def _get_alarm_file_type(filepath):
    extension = os.path.splitext(filepath)[1].lower()
    if extension == ".csv":
        return "csv"
    if extension == ".zip":
        return "zip"
    if extension in (".jsonl", ".json"):
        return "jsonl"
    return None


def _build_file_progress(filepath, file_index=None, total_files=None):
    try:
        total_bytes = os.path.getsize(filepath)
    except OSError:
        total_bytes = 0
    prefix = (
        f"[{file_index}/{total_files}] "
        if file_index is not None and total_files is not None
        else ""
    )
    label = f"{prefix}读取文件 {os.path.basename(filepath)}"
    print(f"⏳ {label}...")
    return ProgressBar(total_bytes, label)


def _build_member_progress(label, total_bytes):
    print(f"⏳ {label}...")
    return ProgressBar(total_bytes, label)


def stream_jsonl_alarms(filepath, show_progress=False, file_index=None, total_files=None):
    try:
        with open(filepath, "r", encoding="utf-8") as file_obj:
            progress = (
                _build_file_progress(filepath, file_index, total_files)
                if show_progress
                else None
            )
            try:
                while True:
                    line = file_obj.readline()
                    if not line:
                        break
                    if progress is not None:
                        progress.set(file_obj.tell())
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            finally:
                if progress is not None:
                    progress.close()
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")


def _stream_jsonl_from_text(text_stream, progress=None, progress_getter=None):
    for line in text_stream:
        if progress is not None and progress_getter is not None:
            progress.set(progress_getter())
        line = line.strip()
        if line:
            yield json.loads(line)


def stream_csv_alarms(filepath, show_progress=False, file_index=None, total_files=None):
    try:
        with open(filepath, "r", encoding="utf-8-sig", newline="") as file_obj:
            progress = (
                _build_file_progress(filepath, file_index, total_files)
                if show_progress
                else None
            )
            try:
                for row in csv.DictReader(file_obj):
                    if progress is not None:
                        progress.set(file_obj.buffer.tell())
                    if row and any(
                        str(value).strip()
                        for value in row.values()
                        if value is not None
                    ):
                        yield row
            finally:
                if progress is not None:
                    progress.close()
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")


def _stream_csv_from_text(text_stream, progress=None, progress_getter=None):
    for row in csv.DictReader(text_stream):
        if progress is not None and progress_getter is not None:
            progress.set(progress_getter())
        if row and any(
            str(value).strip()
            for value in row.values()
            if value is not None
        ):
            yield row


def stream_zip_alarms(filepath, show_progress=False, file_index=None, total_files=None):
    try:
        with zipfile.ZipFile(filepath, "r") as zip_file:
            members = [
                info
                for info in sorted(zip_file.infolist(), key=lambda item: item.filename)
                if not info.is_dir()
                and _get_alarm_file_type(info.filename) in ("csv", "jsonl")
            ]
            if not members:
                print(f"⚠️ 压缩包内没有可读取的告警文件: {filepath}")
                return

            for member in members:
                yield from _stream_zip_member(
                    zip_file, member, filepath, show_progress,
                    file_index, total_files,
                )
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")
    except zipfile.BadZipFile:
        print(f"❌ 非法 zip 文件: {filepath}")


def _stream_zip_member(
    zip_file, member, filepath, show_progress, file_index, total_files
):
    """流式读取压缩包内的单个告警文件（CSV/JSONL）。"""
    prefix = (
        f"[{file_index}/{total_files}] "
        if file_index is not None and total_files is not None
        else ""
    )
    label = (
        f"{prefix}读取压缩文件 "
        f"{os.path.basename(filepath)}::{member.filename}"
    )
    progress = (
        _build_member_progress(label, member.file_size)
        if show_progress
        else None
    )
    with zip_file.open(member, "r") as raw:
        try:
            is_csv = _get_alarm_file_type(member.filename) == "csv"
            encoding = "utf-8-sig" if is_csv else "utf-8"
            text_stream = io.TextIOWrapper(raw, encoding=encoding, newline="")
            try:
                stream_from_text = (
                    _stream_csv_from_text if is_csv else _stream_jsonl_from_text
                )
                yield from stream_from_text(
                    text_stream,
                    progress=progress,
                    progress_getter=raw.tell,
                )
            finally:
                text_stream.detach()
        finally:
            if progress is not None:
                progress.close()


def stream_alarm_file(filepath, show_progress=False, file_index=None, total_files=None):
    file_type = _get_alarm_file_type(filepath)
    readers = {
        "csv": stream_csv_alarms,
        "zip": stream_zip_alarms,
        "jsonl": stream_jsonl_alarms,
    }
    reader = readers.get(file_type)
    if reader is None:
        print(f"⚠️ 跳过不支持的文件类型: {filepath}")
        return
    yield from reader(
        filepath,
        show_progress=show_progress,
        file_index=file_index,
        total_files=total_files,
    )


def stream_alarm_inputs(path, show_progress=False):
    """按文件名字典序读取单个文件或目录中的全部告警。"""
    filepaths = list(_iter_alarm_filepaths(path))
    total_files = len(filepaths)
    for index, filepath in enumerate(filepaths, start=1):
        yield from stream_alarm_file(
            filepath,
            show_progress=show_progress,
            file_index=index,
            total_files=total_files,
        )
