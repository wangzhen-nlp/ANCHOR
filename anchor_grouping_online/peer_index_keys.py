"""端口对端索引的 key 归一化。

写索引（tools/build_resource_buffer.py）与读索引（link_peer_index.py）必须用同一套
归一化逻辑，否则 key 会静默对不上，这里集中维护以保证两端一致。
"""


def normalize_ne_key(ne_id):
    return str(ne_id or "").strip().upper()


def normalize_port_key(port_name):
    return str(port_name or "").strip()


def make_key(ne_id, port_name):
    return f"{normalize_ne_key(ne_id)}|{normalize_port_key(port_name)}"
