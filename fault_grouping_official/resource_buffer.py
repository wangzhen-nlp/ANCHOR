"""读取 build_resource_buffer.py 生成的资源缓冲文件（JSONL）。

每行一个 {"resource_type": <type>, "data": <data>} 资源，resource_type 置于行首，
因此可以先按行前缀判断类型、跳过不需要的资源，无需解析整行 data。
"""

import json


# 行首前缀；与 build_resource_buffer._write_resource_line 的写出格式保持一致。
_RESOURCE_TYPE_PREFIX = '{"resource_type":'
_DECODER = json.JSONDecoder()


def _peek_resource_type(line: str):
    """仅解析行首的 resource_type，不解析整行 data；非资源行返回 None。"""
    if not line.startswith(_RESOURCE_TYPE_PREFIX):
        return None
    rest = line[len(_RESOURCE_TYPE_PREFIX):]
    resource_type, _ = _DECODER.raw_decode(rest)
    return resource_type


def load_resource_buffer(path: str, wanted_types) -> dict:
    """从缓冲文件读取指定资源，返回 {resource_type: data}。

    只对命中 wanted_types 的行解析整行 data，其余行按前缀判断后跳过。
    读到全部 wanted_types 即提前结束。缺失任一类型时报错。
    """
    wanted = set(wanted_types)
    result = {}
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            resource_type = _peek_resource_type(line)
            if resource_type not in wanted:
                continue
            result[resource_type] = json.loads(line)["data"]
            if len(result) == len(wanted):
                break
    missing = wanted - result.keys()
    if missing:
        raise SystemExit(
            f"资源缓冲文件 {path} 缺少资源: {', '.join(sorted(missing))}；"
            "请重新运行 build_resource_buffer.py 生成完整缓冲"
        )
    return result
