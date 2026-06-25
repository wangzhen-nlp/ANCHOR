import json

from dataclasses import asdict, dataclass

from topology_tools.extract_site_graph import load_latest_link_records


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
    port_ip: str = ""
    manager_name: str = ""


def _normalize_ne_key(ne_id):
    return str(ne_id or "").strip().upper()


def _normalize_port_key(port_name):
    return str(port_name or "").strip()


def _make_key(ne_id, port_name):
    return f"{_normalize_ne_key(ne_id)}|{_normalize_port_key(port_name)}"


def _get_record_value(record, *field_names):
    for field_name in field_names:
        value = str(record.get(field_name, "") or "").strip()
        if value:
            return value
    return ""


def build_peer_index_from_sys_link(link_input, report_duplicates=False):
    peer_index = {}

    for record in load_latest_link_records(link_input, report_duplicates=report_duplicates):
        a_ne = _get_record_value(record, "a_end_ne_nativeId", "a_end_ne_nativeId(')")
        z_ne = _get_record_value(record, "z_end_ne_nativeId", "z_end_ne_nativeId(')")
        a_port = _get_record_value(record, "a_end_port_name")
        z_port = _get_record_value(record, "z_end_port_name")
        if not (a_ne and z_ne and a_port and z_port):
            continue

        peer_index[_make_key(a_ne, a_port)] = PeerDevice(
            ne_native_id=_normalize_ne_key(z_ne),
            port_name=z_port,
            port_ip=_get_record_value(record, "z_end_port_ip"),
            manager_name=_get_record_value(record, "z_end_ne_manager_name"),
        )
        peer_index[_make_key(z_ne, z_port)] = PeerDevice(
            ne_native_id=_normalize_ne_key(a_ne),
            port_name=a_port,
            port_ip=_get_record_value(record, "a_end_port_ip"),
            manager_name=_get_record_value(record, "a_end_ne_manager_name"),
        )

    return peer_index


def save_peer_index(peer_index, output_path):
    data = {
        key: asdict(value) if isinstance(value, PeerDevice) else dict(value)
        for key, value in peer_index.items()
    }
    with open(output_path, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2, sort_keys=True)


def load_peer_index(path):
    with open(path, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    return {
        key: PeerDevice(**value) if isinstance(value, dict) else value
        for key, value in data.items()
    }


def resolve_link_alarm_endpoints(alarm_info, alarm_source=""):
    """从 link 告警信息解析本端/对端设备。

    本端设备来自告警源，本端端口来自物理端口；对端通过 peer_index 查询。
    """
    return resolve_link_alarm_endpoints_from_peer_index(
        alarm_info,
        peer_index=None,
        alarm_source=alarm_source,
    )


def resolve_link_alarm_endpoints_from_peer_index(alarm_info, peer_index=None, alarm_source=""):
    alarm_info = alarm_info if isinstance(alarm_info, dict) else {}
    local_ne = str(alarm_source or alarm_info.get("告警源", "") or "").strip()
    local_port = str(alarm_info.get("物理端口", "") or "").strip()
    if not peer_index or not local_ne or not local_port:
        return LinkAlarmEndpoints(local_ne=local_ne, local_port=local_port)

    peer = peer_index.get(_make_key(local_ne, local_port))
    if peer is None:
        return LinkAlarmEndpoints(local_ne=local_ne, local_port=local_port)
    if isinstance(peer, dict):
        peer = PeerDevice(**peer)
    return LinkAlarmEndpoints(
        local_ne=_normalize_ne_key(local_ne),
        local_port=local_port,
        remote_ne=peer.ne_native_id,
        remote_port=peer.port_name,
    )
