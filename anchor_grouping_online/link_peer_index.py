"""构建端口对端索引，并解析链路告警的本端与对端。"""

from dataclasses import dataclass

from anchor_grouping_online.alarm_events.generator import port_vid_from_extendedattr


@dataclass(frozen=True)
class LinkAlarmEndpoints:
    local_ne: str = ""
    local_port: str = ""
    remote_ne: str = ""
    remote_port: str = ""


@dataclass(frozen=True)
class PeerDevice:
    ne_native_id: str
    port_vid: str = ""


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
    local_port_vid = port_vid_from_extendedattr(alarm_info.get("extendedattr", ""))
    if not peer_index:
        return LinkAlarmEndpoints(
            local_ne=local_ne,
            local_port=local_port_vid,
        )

    if not local_port_vid:
        return LinkAlarmEndpoints(
            local_ne=local_ne,
            local_port="",
        )

    peer = peer_index.get(local_port_vid)
    if peer is None:
        return LinkAlarmEndpoints(
            local_ne=local_ne,
            local_port=local_port_vid,
        )
    if isinstance(peer, dict):
        peer = PeerDevice(**peer)
    return LinkAlarmEndpoints(
        local_ne=local_ne,
        local_port=local_port_vid,
        remote_ne=peer.ne_native_id,
        remote_port=peer.port_vid,
    )
