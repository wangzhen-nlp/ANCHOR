import json
from argparse import ArgumentParser
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, resource_display
from ticket_recall.ticket_recall_utils import load_ne_graph_data, normalize_text


def _extract_domain(ne_info):
    if not isinstance(ne_info, dict):
        return ""
    return (
        normalize_text(ne_info.get("domain", ""))
        or normalize_text(ne_info.get("Domain", ""))
        or normalize_text(ne_info.get("DOMAIN", ""))
    ).upper()


def _extract_site_id(ne_info):
    if not isinstance(ne_info, dict):
        return ""
    return normalize_text(ne_info.get("site_id", ""))


def _extract_site_name(ne_info):
    if not isinstance(ne_info, dict):
        return ""
    return (
        normalize_text(ne_info.get("site_name", ""))
        or normalize_text(ne_info.get("siteName", ""))
        or normalize_text(ne_info.get("name", ""))
    )


def collect_transmission_isolated_sites(ne_graph_data):
    site_to_transmission_nes = defaultdict(list)
    site_to_data_nes = defaultdict(list)
    site_to_name = {}
    site_to_any_external_neighbor_sites = defaultdict(set)
    site_has_unknown_neighbors = defaultdict(bool)

    if not isinstance(ne_graph_data, dict):
        return []

    for raw_ne_id, ne_info in ne_graph_data.items():
        if not isinstance(ne_info, dict):
            continue

        ne_id = normalize_text(raw_ne_id)
        site_id = _extract_site_id(ne_info)
        if not ne_id or not site_id:
            continue

        domain = _extract_domain(ne_info)
        if domain == "TRANSMISSION":
            site_to_transmission_nes[site_id].append(ne_id)
        elif domain == "DATA":
            site_to_data_nes[site_id].append(ne_id)
        if site_id not in site_to_name:
            site_name = _extract_site_name(ne_info)
            if site_name:
                site_to_name[site_id] = site_name

        links = ne_info.get("link", {})
        if not isinstance(links, dict):
            continue

        for raw_neighbor_id in links.keys():
            neighbor_id = normalize_text(raw_neighbor_id)
            if not neighbor_id or neighbor_id == ne_id:
                continue

            neighbor_info = ne_graph_data.get(raw_neighbor_id)
            if neighbor_info is None:
                neighbor_info = ne_graph_data.get(neighbor_id)
            neighbor_site_id = _extract_site_id(neighbor_info)

            if not neighbor_site_id:
                site_has_unknown_neighbors[site_id] = True
                continue

            if neighbor_site_id != site_id:
                site_to_any_external_neighbor_sites[site_id].add(neighbor_site_id)

    isolated_sites = []
    for site_id, transmission_ne_ids in site_to_transmission_nes.items():
        external_neighbor_sites = sorted(site_to_any_external_neighbor_sites.get(site_id, set()))
        if (
            external_neighbor_sites
            or site_has_unknown_neighbors.get(site_id, False)
        ):
            continue

        isolated_sites.append(
            {
                "site_id": site_id,
                "site_name": site_to_name.get(site_id, ""),
                "transmission_ne_count": len(transmission_ne_ids),
                "transmission_ne_ids": sorted(transmission_ne_ids),
                "data_ne_count": len(site_to_data_nes.get(site_id, [])),
                "data_ne_ids": sorted(site_to_data_nes.get(site_id, [])),
                "external_neighbor_site_count": 0,
                "external_neighbor_sites": [],
            }
        )

    isolated_sites.sort(
        key=lambda item: (
            -item.get("transmission_ne_count", 0),
            item.get("site_id", ""),
        )
    )
    return isolated_sites


def main():
    parser = ArgumentParser(
        description="从 ne_graph.json 中筛出：站点内有 Transmission 设备，且站内任意设备都不存在跨站连接，同时也不存在站点未知的邻居连接的站点"
    )
    parser.add_argument(
        "ne_graph",
        nargs="?",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件路径，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="transmission_isolated_sites.json",
        help="输出 JSON 文件，默认: transmission_isolated_sites.json",
    )
    parser.add_argument(
        "--site-id-only",
        action="store_true",
        help="仅输出站点ID列表，而不是详细对象列表",
    )
    args = parser.parse_args()

    ne_graph_data = load_ne_graph_data(args.ne_graph)
    if not ne_graph_data:
        raise ValueError(f"未能从 {args.ne_graph} 读取有效的 ne_graph 数据")

    isolated_sites = collect_transmission_isolated_sites(ne_graph_data)

    if args.site_id_only:
        output_payload = [item["site_id"] for item in isolated_sites]
    else:
        output_payload = isolated_sites

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    total_sites = len(
        {
            _extract_site_id(ne_info)
            for ne_info in ne_graph_data.values()
            if isinstance(ne_info, dict) and _extract_site_id(ne_info)
        }
    )
    transmission_site_set = {
        _extract_site_id(ne_info)
        for ne_info in ne_graph_data.values()
        if isinstance(ne_info, dict)
        and _extract_site_id(ne_info)
        and _extract_domain(ne_info) == "TRANSMISSION"
    }

    print(f"总站点数: {total_sites}")
    print(f"包含 Transmission 设备的站点数: {len(transmission_site_set)}")
    print(f"筛出的站点数: {len(isolated_sites)}")
    print(f"结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
