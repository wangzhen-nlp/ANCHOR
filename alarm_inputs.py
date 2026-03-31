import csv
import io
import json
import os
import zipfile

from progress_utils import ProgressBar


def _iter_alarm_filepaths(path):
    if os.path.isdir(path):
        for filename in sorted(os.listdir(path)):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                yield filepath
        return

    yield path


def list_alarm_filepaths(path):
    return list(_iter_alarm_filepaths(path))


def _get_alarm_file_type(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        return 'csv'
    if ext == '.zip':
        return 'zip'
    if ext in ('.jsonl', '.json'):
        return 'jsonl'
    return None


def _build_file_progress(filepath, file_index=None, total_files=None):
    try:
        total_bytes = os.path.getsize(filepath)
    except OSError:
        total_bytes = 0
    prefix = ""
    if file_index is not None and total_files is not None:
        prefix = f"[{file_index}/{total_files}] "
    label = f"{prefix}读取文件 {os.path.basename(filepath)}"
    print(f"⏳ {label}...")
    return ProgressBar(total_bytes, label)


def _build_member_progress(label, total_bytes):
    print(f"⏳ {label}...")
    return ProgressBar(total_bytes, label)


def stream_jsonl_alarms(filepath, show_progress=False, file_index=None, total_files=None):
    """
    使用生成器 (yield) 逐行读取流式告警文件。
    这种在线/流式 (Online/Streaming) 处理方式确保了极低的常驻内存。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            progress = _build_file_progress(filepath, file_index, total_files) if show_progress else None
            try:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if progress is not None:
                        progress.set(f.tell())
                    line = line.strip()
                    if not line:
                        continue
                    # 解析单行 JSON 并直接推入下游
                    yield json.loads(line)
            finally:
                if progress is not None:
                    progress.close()
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")


def _stream_jsonl_from_text(text_stream, progress=None, progress_getter=None):
    while True:
        line = text_stream.readline()
        if not line:
            break
        if progress is not None and progress_getter is not None:
            progress.set(progress_getter())
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def stream_csv_alarms(filepath, show_progress=False, file_index=None, total_files=None):
    """
    使用生成器 (yield) 逐行读取 CSV 告警文件。
    每行会被解析为 dict，以便与 JSONL 输入保持一致。
    """
    try:
        with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
            progress = _build_file_progress(filepath, file_index, total_files) if show_progress else None
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    if progress is not None:
                        progress.set(f.buffer.tell())
                    if not row:
                        continue
                    if not any(str(v).strip() for v in row.values() if v is not None):
                        continue
                    yield row
            finally:
                if progress is not None:
                    progress.close()
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")


def _stream_csv_from_text(text_stream, progress=None, progress_getter=None):
    reader = csv.DictReader(text_stream)
    for row in reader:
        if progress is not None and progress_getter is not None:
            progress.set(progress_getter())
        if not row:
            continue
        if not any(str(v).strip() for v in row.values() if v is not None):
            continue
        yield row


def stream_zip_alarms(filepath, show_progress=False, file_index=None, total_files=None):
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            members = [
                info for info in sorted(zf.infolist(), key=lambda info: info.filename)
                if not info.is_dir() and _get_alarm_file_type(info.filename) in ('csv', 'jsonl')
            ]
            if not members:
                print(f"⚠️ 压缩包内没有可读取的告警文件: {filepath}")
                return

            for member in members:
                prefix = ""
                if file_index is not None and total_files is not None:
                    prefix = f"[{file_index}/{total_files}] "
                label = f"{prefix}读取压缩文件 {os.path.basename(filepath)}::{member.filename}"
                progress = _build_member_progress(label, member.file_size) if show_progress else None
                with zf.open(member, 'r') as raw:
                    try:
                        member_type = _get_alarm_file_type(member.filename)
                        if member_type == 'csv':
                            text_stream = io.TextIOWrapper(raw, encoding='utf-8-sig', newline='')
                            try:
                                yield from _stream_csv_from_text(
                                    text_stream,
                                    progress=progress,
                                    progress_getter=raw.tell
                                )
                            finally:
                                text_stream.detach()
                        else:
                            text_stream = io.TextIOWrapper(raw, encoding='utf-8')
                            try:
                                yield from _stream_jsonl_from_text(
                                    text_stream,
                                    progress=progress,
                                    progress_getter=raw.tell
                                )
                            finally:
                                text_stream.detach()
                    finally:
                        if progress is not None:
                            progress.close()
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")
    except zipfile.BadZipFile:
        print(f"❌ 非法 zip 文件: {filepath}")


def stream_alarm_file(filepath, show_progress=False, file_index=None, total_files=None):
    """
    根据文件扩展名自动选择合适的解析器。
    当前支持 JSONL 和 CSV。
    """
    file_type = _get_alarm_file_type(filepath)
    if file_type == 'csv':
        yield from stream_csv_alarms(
            filepath,
            show_progress=show_progress,
            file_index=file_index,
            total_files=total_files
        )
        return
    if file_type == 'zip':
        yield from stream_zip_alarms(
            filepath,
            show_progress=show_progress,
            file_index=file_index,
            total_files=total_files
        )
        return
    if file_type == 'jsonl':
        yield from stream_jsonl_alarms(
            filepath,
            show_progress=show_progress,
            file_index=file_index,
            total_files=total_files
        )
        return
    print(f"⚠️ 跳过不支持的文件类型: {filepath}")


def stream_alarm_inputs(path, show_progress=False):
    """
    支持单文件或目录输入。
    如果是目录，则按文件名字典序依次读取目录下的普通文件。
    当前支持 JSONL、CSV，以及包含它们的 ZIP 压缩文件。
    """
    filepaths = list_alarm_filepaths(path)
    total_files = len(filepaths)
    for idx, filepath in enumerate(filepaths, start=1):
        yield from stream_alarm_file(
            filepath,
            show_progress=show_progress,
            file_index=idx,
            total_files=total_files
        )


def load_site_graph(site_graph_file: str) -> set:
    """加载站点图中的所有站点（key + value）"""
    with open(site_graph_file, 'r', encoding='utf-8') as f:
        site_graph = json.load(f)

    sites = set(site_graph.keys())
    for _, connected_sites in site_graph.items():
        if isinstance(connected_sites, list):
            sites.update(connected_sites)
        elif isinstance(connected_sites, dict):
            sites.update(connected_sites.keys())

    return sites


def build_ne_to_site_map(ne_graph_file: str) -> dict:
    """构建 ne -> site_id 映射"""
    with open(ne_graph_file, 'r', encoding='utf-8') as f:
        ne_graph = json.load(f)

    ne_to_site = {}
    for ne_name, ne_info in ne_graph.items():
        site_id = ne_info.get('site_id', '')
        site_name = ne_info.get('site_name', '')
        if site_id and site_name:
            ne_to_site[ne_name] = site_id

    return ne_to_site
