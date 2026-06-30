"""加载端口对端索引，并解析链路告警的本端与对端。"""

import json

from dataclasses import dataclass

from fault_grouping_official.peer_index_keys import make_key, normalize_ne_key


@dataclass(frozen=True)
class LinkAlarmEndpoints:
    local_ne: str = ""
    local_port: str = ""
    remote_ne: str = ""
    remote_port: str = ""


@dataclass(frozen=True)
class PeerDevice:
    ne_native_id: str
    port_name: str = ""


def load_peer_index(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return build_peer_index(data)


def build_peer_index(data):
    return {
        key: PeerDevice(**value) if isinstance(value, dict) else value
        for key, value in data.items()
    }


def resolve_link_alarm_endpoints_from_peer_index(
    alarm_info,
    peer_index=None,
    alarm_source="",
):
    alarm_info = alarm_info if isinstance(alarm_info, dict) else {}
    local_ne = str(alarm_source or alarm_info.get("告警源", "") or "").strip()
    local_port = str(alarm_info.get("物理端口名称", "") or "").strip()
    if not peer_index or not local_ne or not local_port:
        return LinkAlarmEndpoints(local_ne=local_ne, local_port=local_port)

    peer = peer_index.get(make_key(local_ne, local_port))
    if peer is None:
        return LinkAlarmEndpoints(local_ne=local_ne, local_port=local_port)
    if isinstance(peer, dict):
        peer = PeerDevice(**peer)
    return LinkAlarmEndpoints(
        local_ne=normalize_ne_key(local_ne),
        local_port=local_port,
        remote_ne=peer.ne_native_id,
        remote_port=peer.port_name,
    )
