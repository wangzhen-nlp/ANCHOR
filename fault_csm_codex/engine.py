import collections
import copy
import heapq
import uuid

from collections.abc import Iterable
from datetime import datetime

from alarm_tools.alarm_types import CRITICAL_ALARMS, POWER_ALARMS
from fault_grouping.temporal_engine.utils import build_pattern_adj


SUPPORTED_ALGORITHMS = ("incisomatch", "sjtree", "graphflow", "iedyn", "turboflux", "symbi")


def normalize_window(edge_window):
    if isinstance(edge_window, dict):
        return (
            float(edge_window.get("before_sec", edge_window.get("backward_sec", 0))),
            float(edge_window.get("after_sec", edge_window.get("forward_sec", 0))),
        )
    win = float(edge_window or 0)
    return win, win


def normalize_directions(direction):
    if isinstance(direction, str):
        text = direction.strip()
        return (text,) if text else ("downstream",)
    if isinstance(direction, Iterable):
        result = []
        seen = set()
        for item in direction:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return tuple(result) if result else ("downstream",)
    return (str(direction).strip() or "downstream",)


class ActiveAlarmIndex:
    """Active alarm vertices and HAS_ALARM dynamic edges.

    In the CSM model used here, an alarm occurrence inserts an alarm vertex and
    a dynamic ``site --HAS_ALARM--> alarm`` edge. A clear event deletes that
    edge/vertex. Rule predicates query these currently active alarm edges.
    """

    def __init__(self, alarm_source_domain_map=None):
        self.by_site = collections.defaultdict(list)
        self.by_eid = {}
        self.expire_heap = []
        self.alarm_source_domain_map = alarm_source_domain_map or {}
        self.change_seq = 0

    def add(self, site_id, alarm_type, ts, eid, alarm_source="", expire_ts=None):
        event = {
            "node": site_id,
            "ts": ts,
            "eid": eid,
            "alarm": alarm_type,
            "alarm_source": alarm_source,
            "alarm_source_domain": self.alarm_source_domain_map.get(alarm_source, ""),
        }
        if expire_ts is not None:
            event["expire_ts"] = float(expire_ts)
        self.by_site[site_id].append(event)
        self.by_eid[eid] = event
        if expire_ts is not None:
            heapq.heappush(self.expire_heap, (float(expire_ts), eid))
        self.change_seq += 1
        return event

    def remove(self, eid):
        event = self.by_eid.pop(eid, None)
        if not event:
            return None
        site_events = self.by_site.get(event["node"])
        if site_events:
            self.by_site[event["node"]] = [item for item in site_events if item.get("eid") != eid]
            if not self.by_site[event["node"]]:
                self.by_site.pop(event["node"], None)
        self.change_seq += 1
        return event

    def site_has_active_alarm(self, site_id):
        return bool(self.by_site.get(site_id))

    def prune_site(self, site_id, current_ts, ttl):
        events = self.by_site.get(site_id)
        if not events:
            return []
        cutoff = current_ts - ttl
        kept = []
        removed = []
        for event in events:
            if event["ts"] < cutoff:
                self.by_eid.pop(event.get("eid"), None)
                removed.append(event)
            else:
                kept.append(event)
        if kept:
            self.by_site[site_id] = kept
        else:
            self.by_site.pop(site_id, None)
        if removed:
            self.change_seq += 1
        return removed

    def expire_until(self, current_ts):
        expired = []
        while self.expire_heap and self.expire_heap[0][0] <= current_ts:
            expire_ts, eid = heapq.heappop(self.expire_heap)
            event = self.by_eid.get(eid)
            if not event or event.get("expire_ts") != expire_ts:
                continue
            removed = self.remove(eid)
            if removed:
                expired.append(removed)
        return expired

    def events_in_window(self, site_id, reference_ts, edge_window):
        before_sec, after_sec = normalize_window(edge_window)
        start_ts = reference_ts - before_sec
        end_ts = reference_ts + after_sec
        return [
            event
            for event in self.by_site.get(site_id, ())
            if start_ts <= event["ts"] <= end_ts
        ]


class SitePredicate:
    def __init__(self, site_domain_map, active_alarm_index):
        self.site_domain_map = site_domain_map
        self.active_alarm_index = active_alarm_index

    @staticmethod
    def has_domain(domain_info, domain):
        if isinstance(domain_info, dict):
            if domain not in domain_info:
                return False
            value = domain_info.get(domain)
            if isinstance(value, (int, float)):
                return value > 0
            if isinstance(value, str):
                return value.strip() not in ("", "0")
            if isinstance(value, (list, tuple, set, dict)):
                return len(value) > 0
            return bool(value)
        if isinstance(domain_info, (list, tuple, set)):
            return domain in domain_info
        if isinstance(domain_info, str):
            return str(domain).strip().lower() == domain_info.strip().lower()
        return False

    @classmethod
    def match_site_rule(cls, domain_info, site_rule):
        include = site_rule.get("include", [])
        exclude = site_rule.get("exclude", [])
        return all(cls.has_domain(domain_info, item) for item in include) and all(
            not cls.has_domain(domain_info, item) for item in exclude
        )

    def matches_structure(self, site_id, node_config):
        domain_info = self.site_domain_map.get(site_id, {})
        site_rules = node_config.get("site_rules")
        if site_rules and not any(self.match_site_rule(domain_info, rule) for rule in site_rules):
            return False
        if node_config.get("type", "primitive") == "compound":
            return any(self.matches_structure(site_id, pattern) for pattern in node_config.get("patterns", []))
        return True

    def resolve_expected_alarms(self, site_id, node_config):
        domain_info = self.site_domain_map.get(site_id, {})
        site_rules = node_config.get("site_rules")
        if site_rules:
            for rule in site_rules:
                if self.match_site_rule(domain_info, rule):
                    return rule.get("expected_alarms")
            return None
        return node_config.get("expected_alarms", "ANY")

    @classmethod
    def _normalize_domain_filter(cls, domain_filter):
        if domain_filter is None:
            return None
        if isinstance(domain_filter, str):
            return [domain_filter]
        if isinstance(domain_filter, Iterable):
            return list(domain_filter)
        return None

    @classmethod
    def matches_source_domain(cls, event, domain_filter):
        domains = cls._normalize_domain_filter(domain_filter)
        if domains is None:
            return domain_filter is None
        source_domain = event.get("alarm_source_domain", "")
        return any(cls.has_domain(source_domain, domain) for domain in domains)

    @classmethod
    def filter_events(cls, events, alarms, source_domains=None):
        return [
            event
            for event in events
            if event.get("alarm") in alarms and cls.matches_source_domain(event, source_domains)
        ]

    def validate(self, site_id, node_config, reference_ts, edge_window):
        if not self.matches_structure(site_id, node_config):
            return False, []

        node_type = node_config.get("type", "primitive")
        if node_type == "compound":
            min_count = int(node_config.get("min_count", 1))
            matched = 0
            events = []
            for pattern in node_config.get("patterns", []):
                valid, pattern_events = self.validate(site_id, pattern, reference_ts, edge_window)
                if valid:
                    matched += 1
                    events.extend(pattern_events)
            return matched >= min_count, self._dedupe_events(events)

        expected = self.resolve_expected_alarms(site_id, node_config)
        if expected is None:
            return False, []
        events_in_window = self.active_alarm_index.events_in_window(site_id, reference_ts, edge_window)

        if expected == "ANY":
            return True, events_in_window
        if expected == "NONE":
            return not any(event.get("alarm") in CRITICAL_ALARMS for event in events_in_window), []
        if isinstance(expected, dict):
            forbidden_alarms = expected.get("forbidden_alarms")
            forbidden_domains = expected.get("forbidden_alarm_source_domains")
            if isinstance(forbidden_alarms, Iterable) and not isinstance(forbidden_alarms, str):
                if any(
                    self.matches_source_domain(event, forbidden_domains)
                    for event in events_in_window
                    if event.get("alarm") in forbidden_alarms
                ):
                    return False, []

            collected = []
            required_alarms = expected.get("required_alarms")
            if isinstance(required_alarms, Iterable) and not isinstance(required_alarms, str):
                required_events = self.filter_events(
                    events_in_window,
                    required_alarms,
                    expected.get("required_alarm_source_domains"),
                )
                if not required_events:
                    return False, []
                collected.extend(required_events)
            elif required_alarms is not None:
                return False, []

            optional_alarms = expected.get("optional_alarms")
            if isinstance(optional_alarms, Iterable) and not isinstance(optional_alarms, str):
                collected.extend(
                    self.filter_events(
                        events_in_window,
                        optional_alarms,
                        expected.get("optional_alarm_source_domains"),
                    )
                )
            elif optional_alarms is not None:
                return False, []

            if collected:
                return True, self._dedupe_events(collected)
            return bool(optional_alarms is not None or forbidden_alarms is not None), []

        if isinstance(expected, Iterable) and not isinstance(expected, str):
            valid_events = [event for event in events_in_window if event.get("alarm") in expected]
            return bool(valid_events), valid_events
        return False, []

    @staticmethod
    def _dedupe_events(events):
        result = []
        seen = set()
        for event in events:
            key = event.get("eid") or (event.get("node"), event.get("ts"), event.get("alarm"), event.get("alarm_source"))
            if key in seen:
                continue
            seen.add(key)
            result.append(event)
        return result


class TopologyIndex:
    def __init__(self, topo_downstream_map, site_chain_index=None):
        self.down = {
            str(site): [str(item) for item in downstreams]
            for site, downstreams in topo_downstream_map.items()
            if isinstance(downstreams, list)
        }
        self.up = collections.defaultdict(list)
        for up_site, down_sites in self.down.items():
            for down_site in down_sites:
                self.up[down_site].append(up_site)
        self.site_chain_index = site_chain_index or {}
        self.cache = {}

    def neighbors(self, site_id, direction):
        if direction == "self":
            return (site_id,)
        if direction == "upstream":
            return tuple(self.up.get(site_id, ()))
        if direction == "downstream":
            return tuple(self.down.get(site_id, ()))
        if direction == "either":
            return tuple(dict.fromkeys([*self.up.get(site_id, ()), *self.down.get(site_id, ())]))
        if direction in {"bidirection", "bidirectional"}:
            return tuple(sorted(set(self.up.get(site_id, ())) & set(self.down.get(site_id, ()))))
        return tuple(self.down.get(site_id, ()))

    def traverse(self, start_site, direction, max_hops=None):
        directions = normalize_directions(direction)
        if len(directions) > 1:
            merged = {}
            for single_direction in directions:
                for site_id, hop in self.traverse(start_site, single_direction, max_hops=max_hops).items():
                    if site_id not in merged or hop < merged[site_id]:
                        merged[site_id] = hop
            return merged

        direction = directions[0]
        cache_key = (start_site, direction, max_hops)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if direction == "self":
            result = {start_site: 0}
            self.cache[cache_key] = result
            return result

        precomputed = self._site_chain_candidates(start_site, direction, max_hops)
        if precomputed is not None:
            self.cache[cache_key] = precomputed
            return precomputed

        result = {}
        visited = {start_site}
        queue = collections.deque([(start_site, 0)])
        while queue:
            site_id, hop = queue.popleft()
            if hop > 0:
                result[site_id] = hop
            if max_hops is not None and hop >= max_hops:
                continue
            for neighbor in self.neighbors(site_id, direction):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, hop + 1))

        self.cache[cache_key] = result
        return result

    def _site_chain_candidates(self, start_site, direction, max_hops):
        info = self.site_chain_index.get(str(start_site or "").strip()) if self.site_chain_index else None
        if not isinstance(info, dict):
            return None
        result = {}

        def add(site_id, hop):
            site_id = str(site_id or "").strip()
            if not site_id or site_id == start_site:
                return
            try:
                hop = int(hop)
            except (TypeError, ValueError):
                return
            if hop <= 0 or (max_hops is not None and hop > max_hops):
                return
            if site_id not in result or hop < result[site_id]:
                result[site_id] = hop

        if direction == "downstream":
            for site_id, hop in (info.get("downstream_site_hops") or {}).items():
                add(site_id, hop)
            return result
        if direction == "upstream":
            for site_id, hop in (info.get("upstream_site_hops") or {}).items():
                add(site_id, hop)
            return result
        if direction in {"bidirection", "bidirectional"}:
            for site_id in info.get("bidirectional_sites") or []:
                add(site_id, 1)
            return result
        if direction == "either" and max_hops == 1:
            for site_id, hop in (info.get("downstream_site_hops") or {}).items():
                add(site_id, hop)
            for site_id, hop in (info.get("upstream_site_hops") or {}).items():
                add(site_id, hop)
            for site_id in info.get("bidirectional_sites") or []:
                add(site_id, 1)
            return result
        return None


class RoleSiteIndex:
    """Static role -> site structural candidate index.

    This keeps fault_csm_codex aligned with fault_grouping semantics: topology is
    still the original topology, while role/site structural compatibility is
    indexed once and reused by the incremental matcher.
    """

    def __init__(self, rules, site_domain_map, predicate):
        self.by_rule_role = {}
        self.by_site = collections.defaultdict(set)
        self._build(rules, site_domain_map, predicate)

    def _build(self, rules, site_domain_map, predicate):
        all_sites = [str(site_id) for site_id in site_domain_map]
        for rule_name, plan in rules.items():
            for role, node_config in plan.nodes.items():
                candidates = {
                    site_id
                    for site_id in all_sites
                    if predicate.matches_structure(site_id, node_config)
                }
                self.by_rule_role[(rule_name, role)] = candidates
                for site_id in candidates:
                    self.by_site[site_id].add((rule_name, role))

    def matches(self, rule_name, role, site_id):
        candidates = self.by_rule_role.get((rule_name, role))
        return candidates is None or site_id in candidates

    def event_seed_roles(self, site_id):
        return self.by_site.get(site_id, set())


class DynamicFaultGraph:
    """Dynamic graph adapter for CSM-style fault matching.

    Static vertices/edges:
    - site vertices
    - original topology relations used by rule edges

    Dynamic vertices/edges:
    - alarm vertices
    - HAS_ALARM edges from site vertices to alarm vertices

    A normal alarm event is represented as AddEdge(site, alarm, HAS_ALARM).
    Clear/expiry removes the corresponding dynamic alarm edge.
    """

    def __init__(self, topology, active_alarms):
        self.topology = topology
        self.active_alarms = active_alarms

    def insert_alarm(self, site_id, alarm_type, ts, eid, alarm_source="", expire_ts=None):
        event = self.active_alarms.add(
            site_id,
            alarm_type,
            ts,
            eid,
            alarm_source=alarm_source,
            expire_ts=expire_ts,
        )
        return {
            "alarm_edge": {
                "kind": "HAS_ALARM",
                "src": site_id,
                "dst": event.get("eid"),
                "site": site_id,
                "alarm": event,
                "label": alarm_type,
            },
        }

    def delete_alarm(self, eid):
        event = self.active_alarms.remove(eid)
        if not event:
            return None
        return event

    def expire_until(self, current_ts):
        return self.active_alarms.expire_until(current_ts)

    def prune_site(self, site_id, current_ts, ttl):
        return self.active_alarms.prune_site(site_id, current_ts, ttl)

    def reachable_sites(self, start_site, query_edge):
        return self.topology.traverse(
            start_site,
            query_edge["traverse_dir"],
            query_edge.get("hops"),
        ).items()

    def has_topology_relation(self, source_site, target_site, query_edge):
        return target_site in self.topology.traverse(
            source_site,
            query_edge["traverse_dir"],
            query_edge.get("hops"),
        )


class StreamingIntermediateCache:
    """Algorithm-inspired local intermediate result cache.

    The rule pattern still comes directly from ``rule_config.py``.  This cache
    only stores reusable IVM-style intermediates:
    - static role-compatible topology neighbors, shared by all algorithms;
    - dynamic support/count checks for algorithms that use support pruning.

    It deliberately avoids building all full-graph candidates.  Alarm add,
    clear, and active-time expiry are the only dynamic update points.
    """

    SUPPORT_PRUNING_ALGORITHMS = {"iedyn", "symbi", "turboflux"}

    def __init__(self, topology, role_index, algorithm):
        self.topology = topology
        self.role_index = role_index
        self.algorithm = algorithm
        self.neighbor_cache = {}
        self.support_cache = {}
        self.dynamic_generation = 0

    @property
    def supports_dynamic_pruning(self):
        return self.algorithm in self.SUPPORT_PRUNING_ALGORITHMS

    def invalidate_dynamic(self):
        self.dynamic_generation += 1
        self.support_cache.clear()

    def candidate_neighbors(self, plan, source_role, source_site, target_role, edge):
        key = (
            plan.name,
            source_role,
            target_role,
            source_site,
            self._edge_cache_key(edge),
        )
        if key not in self.neighbor_cache:
            self.neighbor_cache[key] = tuple(
                site_id
                for site_id, _hop in self.topology.traverse(
                    source_site,
                    edge["traverse_dir"],
                    edge.get("hops"),
                ).items()
                if self.role_index.matches(plan.name, target_role, site_id)
            )
        return self.neighbor_cache[key]

    def support_get(self, key):
        return self.support_cache.get((self.dynamic_generation, *key))

    def support_set(self, key, value):
        self.support_cache[(self.dynamic_generation, *key)] = value
        return value

    @staticmethod
    def _edge_cache_key(edge):
        return (
            edge.get("traverse_dir"),
            edge.get("hops"),
            edge.get("win"),
            bool(edge.get("optional")),
            bool(edge.get("dedupe_symmetric_pair")),
        )


class RulePlan:
    def __init__(self, name, config, algorithm="graphflow"):
        self.name = name
        self.config = config
        self.algorithm = algorithm
        self.nodes = config.get("nodes", {})
        self.trigger_role = config["trigger_role"]
        self.pattern_adj = build_pattern_adj(config.get("edges", []))
        self.query_orders = self._build_query_orders()
        self.alarm_query_edges = self._build_alarm_query_edges()
        self.root_roles = tuple(
            role
            for role in self.nodes
            if role not in {edge["target"] for edge in config.get("edges", [])}
        )
        self.roles = tuple(self.nodes.keys())
        self.required_roles = {
            role
            for role, node_config in self.nodes.items()
            if not self._role_is_context_only(node_config)
        }

    def _build_alarm_query_edges(self):
        """Compile role alarm predicates into implicit query HAS_ALARM edges."""
        return {
            role: {
                "source_role": role,
                "alarm_vertex": f"{role}__alarm",
                "edge_label": "HAS_ALARM",
                "node_config": node_config,
            }
            for role, node_config in self.nodes.items()
            if self._node_can_have_alarm_edge(node_config)
        }

    @classmethod
    def _node_can_have_alarm_edge(cls, node_config):
        if node_config.get("type") == "compound":
            return any(cls._node_can_have_alarm_edge(pattern) for pattern in node_config.get("patterns", []))
        site_rules = node_config.get("site_rules") or []
        expected_values = [rule.get("expected_alarms") for rule in site_rules] or [node_config.get("expected_alarms")]
        for expected in expected_values:
            if expected == "ANY":
                return True
            if isinstance(expected, Iterable) and not isinstance(expected, str):
                return True
            if isinstance(expected, dict):
                if expected.get("required_alarms") is not None or expected.get("optional_alarms") is not None:
                    return True
        return False

    def _build_query_orders(self):
        if self.algorithm == "incisomatch":
            return self._build_incisomatch_orders()
        if self.algorithm == "sjtree":
            return self._build_sjtree_orders()
        if self.algorithm == "iedyn":
            return self._build_iedyn_orders()
        if self.algorithm == "symbi":
            return self._build_symbi_orders()
        if self.algorithm == "turboflux":
            return self._build_turboflux_orders()
        return self._build_graphflow_orders()

    def _build_incisomatch_orders(self):
        """Build IncIsoMatch-style DFS orders.

        IncIsoMatch is the lightest IVM baseline: after an updated edge is
        bound, it expands query vertices in deterministic query order, only
        requiring connectivity to the already matched prefix.
        """
        return self._build_orders_from_base_role_order(list(self.nodes), prefer_selective_fallback=False)

    def _build_sjtree_orders(self):
        """Build SJ-Tree-style orders from selective edge decomposition.

        SJ-Tree decomposes a query into edge/subgraph units and joins partial
        matches. In this compact implementation, the equivalent plan orders
        roles by selective query-edge units before using the common backtracker.
        """
        edge_ranked_roles = []
        seen = set()
        ranked_edges = sorted(
            (
                (
                    self._role_selectivity_rank(edge["source_role"]) + self._role_selectivity_rank(edge["role"]),
                    -(len(self.pattern_adj.get(edge["source_role"], ())) + len(self.pattern_adj.get(edge["role"], ()))),
                    edge["source_role"],
                    edge["role"],
                )
                for source_role in self.nodes
                for edge in self.pattern_adj.get(source_role, ())
                if str(source_role) <= str(edge["role"])
            ),
            key=lambda item: item,
        )
        for _score, _degree, source_role, target_role in ranked_edges:
            for role in (source_role, target_role):
                if role not in seen:
                    seen.add(role)
                    edge_ranked_roles.append(role)
        for role in self.nodes:
            if role not in seen:
                edge_ranked_roles.append(role)
        return self._build_orders_from_base_role_order(edge_ranked_roles)

    def _build_iedyn_orders(self):
        """Build IEDyn-style DAG/DCS orders.

        IEDyn uses a query DAG plus incremental candidate support maintenance.
        Here we use the same DAG serialization as SymBi but prefer roles with
        stricter predicates and more already-bound neighbors during expansion.
        """
        serialized_roles = self._query_dag_serialized_roles(prefer_sparse_edge=True)
        ranked_roles = sorted(
            serialized_roles,
            key=lambda role: (self._role_selectivity_rank(role), -len(self.pattern_adj.get(role, ())), role),
        )
        return self._build_orders_from_base_role_order(ranked_roles)

    def _build_graphflow_orders(self):
        orders = {}
        roles = list(self.nodes)
        for start_role in roles:
            for edge in self.pattern_adj.get(start_role, []):
                second_role = edge["role"]
                visited = {start_role, second_role}
                order = [start_role, second_role]
                backward = {
                    start_role: [],
                    second_role: [start_role],
                }
                while len(order) < len(roles):
                    best_role = None
                    best_adjacent = -1
                    best_degree = -1
                    for role in roles:
                        if role in visited:
                            continue
                        adjacent_count = sum(
                            1
                            for adj_edge in self.pattern_adj.get(role, [])
                            if adj_edge["role"] in visited
                        )
                        if adjacent_count <= 0:
                            continue
                        degree = len(self.pattern_adj.get(role, []))
                        if adjacent_count > best_adjacent or (
                            adjacent_count == best_adjacent and degree > best_degree
                        ):
                            best_role = role
                            best_adjacent = adjacent_count
                            best_degree = degree
                    if best_role is None:
                        for role in roles:
                            if role not in visited:
                                best_role = role
                                break
                    visited.add(best_role)
                    order.append(best_role)
                    backward[best_role] = [
                        adj_edge["role"]
                        for adj_edge in self.pattern_adj.get(best_role, [])
                        if adj_edge["role"] in visited and adj_edge["role"] != best_role
                    ]
                orders[(start_role, second_role)] = {
                    "order": tuple(order),
                    "backward": {role: tuple(backward.get(role, ())) for role in roles},
                }
        return orders

    def _build_symbi_orders(self):
        """Build SymBi-like incremental orders from a query DAG.

        The C++ SymBi implementation builds a DAG and then enumerates from the
        update edge using predefined order/backward neighbors. Here we keep that
        structure in Python: query root is selected by edge/role selectivity,
        the DAG is serialized by BFS, and each update-edge order starts with the
        updated edge before following the DAG order.
        """
        serialized_roles = self._query_dag_serialized_roles(prefer_sparse_edge=True)
        return self._build_orders_from_base_role_order(serialized_roles)

    def _build_turboflux_orders(self):
        """Build TurboFlux-like orders.

        TurboFlux chooses a selective query DAG and derives matching orders from
        candidate path counts. Without materializing C++ DCS counters here, we
        approximate the same intent by using a selectivity-sorted DAG order:
        roles with stricter alarm predicates and higher query degree are placed
        earlier once connected to the already matched prefix.
        """
        serialized_roles = self._query_dag_serialized_roles(prefer_sparse_edge=True)
        ranked_roles = sorted(
            serialized_roles,
            key=lambda role: (self._role_selectivity_rank(role), -len(self.pattern_adj.get(role, ()))),
        )
        return self._build_orders_from_base_role_order(ranked_roles)

    def _build_orders_from_base_role_order(self, base_order, prefer_selective_fallback=True):
        orders = {}
        roles = list(self.nodes)
        for start_role in roles:
            for edge in self.pattern_adj.get(start_role, []):
                second_role = edge["role"]
                visited = {start_role, second_role}
                order = [start_role, second_role]
                for role in base_order:
                    if role not in visited and self._role_has_neighbor_in(role, visited):
                        visited.add(role)
                        order.append(role)
                while len(order) < len(roles):
                    best_role = None
                    best_score = None
                    for role in roles:
                        if role in visited:
                            continue
                        connected = self._role_has_neighbor_in(role, visited)
                        if prefer_selective_fallback:
                            score = (
                                0 if connected else 1,
                                self._role_selectivity_rank(role),
                                -len(self.pattern_adj.get(role, ())),
                                role,
                            )
                        else:
                            score = (
                                0 if connected else 1,
                                list(self.nodes).index(role),
                                role,
                            )
                        if best_score is None or score < best_score:
                            best_score = score
                            best_role = role
                    visited.add(best_role)
                    order.append(best_role)
                orders[(start_role, second_role)] = self._order_info_from_order(order)
        return orders

    def _query_dag_serialized_roles(self, prefer_sparse_edge=False):
        roles = list(self.nodes)
        if not roles:
            return []
        root = self._select_query_root(prefer_sparse_edge=prefer_sparse_edge)
        visited = {root}
        order = [root]
        queue = collections.deque([root])
        while queue:
            role = queue.popleft()
            neighbors = sorted(
                (edge["role"] for edge in self.pattern_adj.get(role, ()) if edge["role"] not in visited),
                key=lambda item: (self._role_selectivity_rank(item), -len(self.pattern_adj.get(item, ())), item),
            )
            for neighbor in neighbors:
                visited.add(neighbor)
                order.append(neighbor)
                queue.append(neighbor)
        for role in roles:
            if role not in visited:
                order.append(role)
        return order

    def _select_query_root(self, prefer_sparse_edge=False):
        if not prefer_sparse_edge:
            return min(
                self.nodes,
                key=lambda role: (self._role_selectivity_rank(role), -len(self.pattern_adj.get(role, ())), role),
            )
        best_pair = None
        for source_role in self.nodes:
            for edge in self.pattern_adj.get(source_role, ()):
                target_role = edge["role"]
                pair_score = (
                    self._role_selectivity_rank(source_role) + self._role_selectivity_rank(target_role),
                    -(len(self.pattern_adj.get(source_role, ())) + len(self.pattern_adj.get(target_role, ()))),
                    source_role,
                    target_role,
                )
                if best_pair is None or pair_score < best_pair[0]:
                    best_pair = (pair_score, source_role, target_role)
        if best_pair is None:
            return next(iter(self.nodes))
        _score, source_role, target_role = best_pair
        return min(
            (source_role, target_role),
            key=lambda role: (self._role_selectivity_rank(role), -len(self.pattern_adj.get(role, ())), role),
        )

    def _order_info_from_order(self, order):
        seen = set()
        backward = {}
        for role in order:
            backward[role] = tuple(
                edge["role"]
                for edge in self.pattern_adj.get(role, ())
                if edge["role"] in seen
            )
            seen.add(role)
        return {
            "order": tuple(order),
            "backward": backward,
        }

    def _role_has_neighbor_in(self, role, visited_roles):
        return any(edge["role"] in visited_roles for edge in self.pattern_adj.get(role, ()))

    def _role_selectivity_rank(self, role):
        node_config = self.nodes.get(role, {})
        if node_config.get("type") == "compound":
            ranks = [
                self._node_config_selectivity_rank(pattern)
                for pattern in node_config.get("patterns", [])
            ]
            return min(ranks) if ranks else 50
        return self._node_config_selectivity_rank(node_config)

    @staticmethod
    def _node_config_selectivity_rank(node_config):
        site_rules = node_config.get("site_rules") or []
        expected_values = [rule.get("expected_alarms") for rule in site_rules] or [node_config.get("expected_alarms")]
        best = 50
        for expected in expected_values:
            if isinstance(expected, dict):
                if expected.get("required_alarms") is not None:
                    best = min(best, 0)
                elif expected.get("forbidden_alarms") is not None:
                    best = min(best, 20)
                elif expected.get("optional_alarms") is not None:
                    best = min(best, 30)
            elif isinstance(expected, Iterable) and not isinstance(expected, str):
                best = min(best, 5)
            elif expected == "NONE":
                best = min(best, 25)
            elif expected == "ANY":
                best = min(best, 40)
        return best

    @staticmethod
    def _role_is_context_only(node_config):
        expected = None
        site_rules = node_config.get("site_rules") or []
        if site_rules:
            expected = site_rules[0].get("expected_alarms")
        if node_config.get("type") == "compound":
            return False
        if isinstance(expected, dict):
            return expected.get("required_alarms") is None
        return expected in ("ANY", "NONE")


class TurboFluxDCS:
    """Python implementation of TurboFlux DCS (Dynamic Candidate Support).

    This replicates the C++ TurboFlux algorithm semantics at the structural
    level: query DAG, three-pass DCS construction, CountDownwards path-count
    DP, per-edge matching orders with join_check arrays, and DCS-based
    backtracking.  Incremental maintenance currently does a targeted rebuild
    rather than queue propagation (which keeps the matching engine semantics
    identical while avoiding the complexity of partial invalidation in a
    dynamic predicate model).
    """

    def __init__(self, plan, topology, active_alarms, predicate, role_index=None):
        self.plan = plan
        self.topology = topology
        self.active_alarms = active_alarms
        self.predicate = predicate
        self.role_index = role_index

        self.roles = list(plan.nodes)
        self.pattern_adj = plan.pattern_adj

        # ----- DAG (spanning tree) -----
        self.dag_forwards = {}      # role -> [(child_role, edge_def), ...]
        self.dag_backwards = {}     # role -> [(parent_role, edge_def), ...]
        self.q_root = None
        self.serialized_tree = []
        self.tree_edges = set()     # {(parent, child), ...}
        self.nontree_edges = set()  # {(u, v), ...}

        # ----- DCS (tree edges only, keyed by DAG direction) -----
        # DCS[(parent_role, child_role)][parent_site] = [child_sites]
        self.DCS = {}

        # ----- Status flags -----
        self.d1 = {}   # {role: {site_id: bool}}
        self.d2 = {}   # {role: {site_id: bool}}

        # ----- Counters -----
        self.n1 = {}   # {(parent, child): {parent_site: int}}
        self.np1 = {}  # {role: {site_id: int}}
        self.n2 = {}   # {(parent, child): {parent_site: int}}
        self.nc2 = {}  # {role: {site_id: int}}

        # ----- Matching orders -----
        # orders[(start_role, second_role)] = {
        #     "order": tuple,
        #     "backward": {role: [parent_roles]},
        #     "join_check_vs": {role: [check_roles]},
        #     "join_check_labels": {role: [labels]},
        # }
        self.orders = {}

        # ----- Caches -----
        self._traverse_cache = {}

    # ------------------------------------------------------------------ #
    #  Utility helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _edge_key(src, dst):
        return (src, dst)

    def _get_query_edge(self, src, dst):
        for edge in self.pattern_adj.get(src, []):
            if edge["role"] == dst:
                return edge
        for edge in self.pattern_adj.get(dst, []):
            if edge["role"] == src:
                return edge
        return None

    def _reverse_direction(self, direction):
        if direction == "downstream":
            return "upstream"
        if direction == "upstream":
            return "downstream"
        return direction

    def _topo_reachable(self, from_site, edge):
        if edge is None:
            return {}
        direction = edge.get("traverse_dir", "downstream")
        max_hops = edge.get("hops")
        cache_key = (from_site, direction, max_hops)
        if cache_key not in self._traverse_cache:
            self._traverse_cache[cache_key] = self.topology.traverse(
                from_site, direction, max_hops
            )
        return self._traverse_cache[cache_key]

    def _site_can_reach(self, from_site, to_site, query_edge, from_role, to_role):
        """Return True if from_site can reach to_site w.r.t. the query edge."""
        if query_edge is None:
            return False
        # Query edge direction matches from_role -> to_role
        if (
            query_edge.get("source_role") == from_role
            and query_edge.get("role") == to_role
        ):
            return to_site in self._topo_reachable(from_site, query_edge)
        # Query edge is reversed: to_role -> from_role
        if (
            query_edge.get("source_role") == to_role
            and query_edge.get("role") == from_role
        ):
            rev_edge = dict(query_edge)
            rev_edge["traverse_dir"] = self._reverse_direction(
                query_edge.get("traverse_dir", "downstream")
            )
            return to_site in self._topo_reachable(from_site, rev_edge)
        return False

    def _get_candidate_sites(self, role):
        """Sites that structurally match *role*.

        Context roles in fault_grouping may have no alarms, so DCS candidates
        must not be limited to sites currently present in the active alarm
        index.
        """
        if self.role_index is not None:
            return list(self.role_index.by_rule_role.get((self.plan.name, role), ()))
        node_config = self.plan.nodes[role]
        return [
            site_id
            for site_id in self.predicate.site_domain_map
            if self.predicate.matches_structure(site_id, node_config)
        ]

    def _query_neighbors(self, role):
        nbrs = set()
        for edge in self.pattern_adj.get(role, []):
            nbrs.add(edge["role"])
        for r in self.roles:
            for edge in self.pattern_adj.get(r, []):
                if edge["role"] == role:
                    nbrs.add(r)
        return list(nbrs)

    def _clear_caches(self):
        self._traverse_cache.clear()

    # ------------------------------------------------------------------ #
    #  Public lifecycle
    # ------------------------------------------------------------------ #

    def build(self, eval_cache=None):
        self._clear_caches()
        self._build_dag()
        self._build_dcs(eval_cache=eval_cache)
        self._generate_matching_orders()

    def on_graph_change(self, eval_cache=None):
        """Called after any insert/delete that may change candidate sets."""
        self._clear_caches()
        self._build_dcs(eval_cache=eval_cache)
        self._generate_matching_orders()

    # ------------------------------------------------------------------ #
    #  DAG construction
    # ------------------------------------------------------------------ #

    def _build_dag(self):
        if not self.roles:
            return

        # 1. Edge selectivity scores
        edge_scores = {}
        for src_role in self.roles:
            for edge in self.pattern_adj.get(src_role, []):
                dst_role = edge["role"]
                if str(src_role) <= str(dst_role):
                    pair_score = (
                        self.plan._role_selectivity_rank(src_role)
                        + self.plan._role_selectivity_rank(dst_role),
                        -(
                            len(self.pattern_adj.get(src_role, ()))
                            + len(self.pattern_adj.get(dst_role, ()))
                        ),
                        src_role,
                        dst_role,
                    )
                    edge_scores[(src_role, dst_role)] = pair_score

        # 2. Select root
        if edge_scores:
            best_pair = min(edge_scores.items(), key=lambda x: x[1])
            (src_role, dst_role), _ = best_pair
            self.q_root = min(
                (src_role, dst_role),
                key=lambda r: (
                    self.plan._role_selectivity_rank(r),
                    -len(self.pattern_adj.get(r, ())),
                    r,
                ),
            )
        else:
            self.q_root = self.roles[0]

        # 3. Greedy BFS spanning tree
        visited = {self.q_root}
        self.serialized_tree = [self.q_root]
        queue = collections.deque([self.q_root])

        self.dag_forwards = {role: [] for role in self.roles}
        self.dag_backwards = {role: [] for role in self.roles}
        self.tree_edges = set()

        while queue:
            current = queue.popleft()
            nbrs = []
            seen = set()
            # Outgoing pattern edges
            for edge in self.pattern_adj.get(current, []):
                nbr = edge["role"]
                if nbr not in visited and nbr != current and nbr not in seen:
                    seen.add(nbr)
                    nbrs.append(
                        (
                            (
                                self.plan._role_selectivity_rank(nbr),
                                -len(self.pattern_adj.get(nbr, ())),
                                nbr,
                            ),
                            nbr,
                            edge,
                        )
                    )
            # Incoming pattern edges
            for r in self.roles:
                if r == current or r in visited or r in seen:
                    continue
                for edge in self.pattern_adj.get(r, []):
                    if edge["role"] == current:
                        seen.add(r)
                        nbrs.append(
                            (
                                (
                                    self.plan._role_selectivity_rank(r),
                                    -len(self.pattern_adj.get(r, ())),
                                    r,
                                ),
                                r,
                                edge,
                            )
                        )
                        break

            nbrs.sort(key=lambda x: x[0])
            for _, nbr, edge in nbrs:
                if nbr not in visited:
                    visited.add(nbr)
                    self.serialized_tree.append(nbr)
                    queue.append(nbr)
                    self.dag_forwards[current].append((nbr, edge))
                    self.dag_backwards[nbr].append((current, edge))
                    self.tree_edges.add((current, nbr))

        # Non-tree edges (exclude optional)
        self.nontree_edges = set()
        for src_role in self.roles:
            for edge in self.pattern_adj.get(src_role, []):
                dst_role = edge["role"]
                if edge.get("optional"):
                    continue
                if (src_role, dst_role) not in self.tree_edges and (
                    dst_role,
                    src_role,
                ) not in self.tree_edges:
                    self.nontree_edges.add((src_role, dst_role))

    # ------------------------------------------------------------------ #
    #  Three-pass DCS construction
    # ------------------------------------------------------------------ #

    def _build_dcs(self, eval_cache=None):
        # Reset
        self.DCS = {}
        self.d1 = {role: {} for role in self.roles}
        self.d2 = {role: {} for role in self.roles}
        self.n1 = {}
        self.np1 = {role: {} for role in self.roles}
        self.n2 = {}
        self.nc2 = {role: {} for role in self.roles}

        # Allocate DCS containers for tree edges (DAG direction)
        for parent_role in self.roles:
            for child_role, _edge in self.dag_forwards.get(parent_role, []):
                key = self._edge_key(parent_role, child_role)
                self.DCS[key] = collections.defaultdict(list)
                self.n1[key] = collections.defaultdict(int)
                self.n2[key] = collections.defaultdict(int)

        # Pass 1: top-down
        for role in self.serialized_tree:
            candidates = self._get_candidate_sites(role)
            for site_id in candidates:
                # Parents in DAG
                for parent_role, edge_def in self.dag_backwards.get(role, []):
                    query_edge = self._get_query_edge(parent_role, role)
                    if query_edge is None:
                        continue
                    key = self._edge_key(parent_role, role)
                    for parent_site in self._get_candidate_sites(parent_role):
                        if self._site_can_reach(
                            parent_site, site_id, query_edge, parent_role, role
                        ):
                            if site_id not in self.DCS[key][parent_site]:
                                self.DCS[key][parent_site].append(site_id)
                            # n1 is keyed by child site (site_id), not parent site
                            if self.d1.get(parent_role, {}).get(parent_site, False):
                                self.n1[key][site_id] += 1

                # d1
                num_back = len(self.dag_backwards.get(role, []))
                if num_back == 0:
                    self.d1[role][site_id] = True
                else:
                    sat = 0
                    for parent_role, _ in self.dag_backwards.get(role, []):
                        key = self._edge_key(parent_role, role)
                        if self.n1[key].get(site_id, 0) > 0:
                            sat += 1
                    self.np1[role][site_id] = sat
                    if sat == num_back:
                        self.d1[role][site_id] = True

        # Pass 2: bottom-up
        for role in reversed(self.serialized_tree):
            candidates = self._get_candidate_sites(role)
            for site_id in candidates:
                # Children in DAG
                for child_role, edge_def in self.dag_forwards.get(role, []):
                    query_edge = self._get_query_edge(role, child_role)
                    if query_edge is None:
                        continue
                    key = self._edge_key(role, child_role)
                    for child_site in self._get_candidate_sites(child_role):
                        if self._site_can_reach(
                            site_id, child_site, query_edge, role, child_role
                        ):
                            if child_site not in self.DCS[key][site_id]:
                                self.DCS[key][site_id].append(child_site)
                            # n2 is keyed by parent site (site_id), not child site
                            if self.d2.get(child_role, {}).get(child_site, False):
                                self.n2[key][site_id] += 1

                # d2
                num_fwd = len(self.dag_forwards.get(role, []))
                if num_fwd == 0:
                    if self.d1.get(role, {}).get(site_id, False):
                        self.d2[role][site_id] = True
                else:
                    sat = 0
                    for child_role, _ in self.dag_forwards.get(role, []):
                        key = self._edge_key(role, child_role)
                        if self.n2[key].get(site_id, 0) > 0:
                            sat += 1
                    self.nc2[role][site_id] = sat
                    if sat == num_fwd and self.d1.get(role, {}).get(site_id, False):
                        self.d2[role][site_id] = True

        # Pass 3: top-down modify n2
        for role in self.serialized_tree:
            for parent_role, _ in self.dag_backwards.get(role, []):
                key = self._edge_key(parent_role, role)
                for parent_site, children in self.DCS[key].items():
                    for child_site in children:
                        if self.d2.get(parent_role, {}).get(parent_site, False):
                            self.n2[key][parent_site] += 1

    # ------------------------------------------------------------------ #
    #  CountDownwards DP
    # ------------------------------------------------------------------ #

    def _count_downwards(self, u, v, num_explicit_pathes, num_dp):
        """Recursively count valid downward paths from (query=u, data=v)."""
        for child_role, _edge in self.dag_forwards.get(u, []):
            key = self._edge_key(u, child_role)
            dcs_entry = self.DCS.get(key, {})
            candidates = dcs_entry.get(v, [])

            if child_role not in num_dp:
                num_dp[child_role] = {}

            if v not in num_dp[child_role]:
                cnt = 0
                for v_c in candidates:
                    if self.d2.get(child_role, {}).get(v_c, False):
                        cnt += 1
                        self._count_downwards(child_role, v_c, num_explicit_pathes, num_dp)
                num_dp[child_role][v] = cnt
                num_explicit_pathes[child_role] = num_explicit_pathes.get(child_role, 0) + cnt
            else:
                num_explicit_pathes[child_role] = (
                    num_explicit_pathes.get(child_role, 0) + num_dp[child_role][v]
                )

    # ------------------------------------------------------------------ #
    #  Matching-order generation
    # ------------------------------------------------------------------ #

    def _generate_matching_orders(self):
        self.orders = {}
        if not self.roles:
            return

        # 1. Path counts from root
        num_explicit_pathes = {role: 0 for role in self.roles}
        num_dp = {}
        for v, flag in self.d2.get(self.q_root, {}).items():
            if flag:
                self._count_downwards(self.q_root, v, num_explicit_pathes, num_dp)
        # C++ sets root to a huge number so it is never picked as a leaf
        num_explicit_pathes[self.q_root] = 10 ** 18

        # 2. Leaf-stripping initial_order (leaves first)
        children_count = {
            role: len(self.dag_forwards.get(role, [])) for role in self.roles
        }
        parent_of = {}
        for role in self.roles:
            for child_role, _ in self.dag_forwards.get(role, []):
                parent_of[child_role] = role

        remaining = set(self.roles)
        initial_order = []
        leaves = [r for r in remaining if children_count.get(r, 0) == 0]
        while remaining:
            best_leaf = min(
                leaves,
                key=lambda r: (num_explicit_pathes.get(r, float("inf")), r),
            )
            initial_order.append(best_leaf)
            remaining.remove(best_leaf)
            leaves.remove(best_leaf)
            p = parent_of.get(best_leaf)
            if p and p in remaining:
                children_count[p] -= 1
                if children_count[p] == 0:
                    leaves.append(p)

        # 3. Per-edge orders (fault_csm_codex seeds from any alarm edge)
        for start_role in self.roles:
            for edge in self.pattern_adj.get(start_role, []):
                second_role = edge["role"]
                order, backward, join_vs, join_lbl = self._build_order_for_edge(
                    start_role, second_role, initial_order, parent_of
                )
                self.orders[(start_role, second_role)] = {
                    "order": tuple(order),
                    "backward": backward,
                    "join_check_vs": join_vs,
                    "join_check_labels": join_lbl,
                }

    def _build_order_for_edge(self, start_role, second_role, initial_order, parent_of):
        order = [start_role, second_role]
        visited = {start_role, second_role}
        backward = {start_role: [], second_role: [start_role]}

        # Walk up from second_role toward root
        cur = second_role
        while cur in parent_of:
            p = parent_of[cur]
            if p not in visited:
                order.append(p)
                visited.add(p)
                backward[p] = [
                    nbr for nbr in self._query_neighbors(p) if nbr in visited and nbr != p
                ]
            cur = p

        # Append remaining vertices in initial_order (root-to-leaf sense)
        for role in reversed(initial_order):
            if role not in visited:
                order.append(role)
                visited.add(role)
                backward[role] = [
                    nbr
                    for nbr in self._query_neighbors(role)
                    if nbr in visited and nbr != role
                ]

        # join_check for non-tree edges
        join_check_vs = {role: [] for role in order}
        join_check_labels = {role: [] for role in order}
        for i, role in enumerate(order):
            for j in range(i):
                prev = order[j]
                if (prev, role) in self.nontree_edges or (role, prev) in self.nontree_edges:
                    edge = self._get_query_edge(prev, role)
                    if edge:
                        join_check_vs[role].append(prev)
                        join_check_labels[role].append(
                            edge.get("traverse_dir", "downstream")
                        )

        return order, backward, join_check_vs, join_check_labels

    # ------------------------------------------------------------------ #
    #  FindMatches
    # ------------------------------------------------------------------ #

    def find_matches(
        self,
        order_key,
        mapping,
        visited_sites,
        results,
        trigger_ts,
        eval_cache,
        plan,
        engine,
        max_results=None,
    ):
        """TurboFlux-style backtracking using DCS candidates."""
        order_info = self.orders.get(order_key)
        if order_info is None:
            return

        order = order_info["order"]
        backward = order_info["backward"]
        join_check_vs = order_info["join_check_vs"]
        join_check_labels = order_info["join_check_labels"]

        depth = len(mapping)
        if depth >= len(order):
            match = engine._build_match_from_mapping(plan, mapping, trigger_ts)
            if match is not None and engine._validate_result_constraints(
                plan.config, match
            ):
                results.append(match)
            return

        role = order[depth]
        backward_roles = [r for r in backward.get(role, []) if r in mapping]
        if not backward_roles:
            return

        # Choose the backward role with the smallest DCS candidate set
        best_parent = None
        best_candidates = None
        for parent_role in backward_roles:
            parent_site = mapping[parent_role][0]
            # Tree edge parent_role -> role
            if (parent_role, role) in self.tree_edges:
                key = self._edge_key(parent_role, role)
                candidates = list(self.DCS.get(key, {}).get(parent_site, []))
            # Tree edge role -> parent_role (reversed)
            elif (role, parent_role) in self.tree_edges:
                key = self._edge_key(role, parent_role)
                candidates = []
                for src_site, dst_sites in self.DCS.get(key, {}).items():
                    if parent_site in dst_sites:
                        candidates.append(src_site)
            else:
                # Non-tree edge – fall back to topology traversal
                edge = engine._query_edge(plan, parent_role, role)
                candidates = [
                    site
                    for site, _ in engine.graph.reachable_sites(
                        parent_site, edge
                    )
                ]
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_parent = parent_role
                best_candidates = candidates

        if not best_candidates:
            return

        node_config = plan.nodes[role]
        for candidate_site in best_candidates:
            if candidate_site in visited_sites:
                continue

            # DCS feasibility flag d2
            if not self.d2.get(role, {}).get(candidate_site, False):
                continue

            # Non-tree joinability check
            joinable = True
            for check_role in join_check_vs.get(role, []):
                check_site = mapping[check_role][0]
                edge = engine._query_edge(plan, check_role, role)
                if edge and not engine.graph.has_topology_relation(
                    check_site, candidate_site, edge
                ):
                    joinable = False
                    break
            if not joinable:
                continue

            # SitePredicate validation (time-window dependent, done at match time)
            ref_ts = (
                mapping[best_parent][1][0]["ts"]
                if mapping[best_parent][1]
                else trigger_ts
            )
            qedge = engine._query_edge(plan, best_parent, role)
            win = qedge.get("win", 0) if qedge else 0
            valid, events = engine._validate_cached(
                eval_cache, candidate_site, role, node_config, ref_ts, win
            )
            if not valid:
                continue

            mapping[role] = (candidate_site, events)
            visited_sites.add(candidate_site)
            self.find_matches(
                order_key,
                mapping,
                visited_sites,
                results,
                trigger_ts,
                eval_cache,
                plan,
                engine,
                max_results,
            )
            visited_sites.remove(candidate_site)
            del mapping[role]

            if max_results is not None and len(results) >= max_results:
                return


class FaultCSMEngine:
    """CSM-style incremental matcher for fault grouping.

    Each alarm insertion mutates the data graph by inserting a HAS_ALARM edge.
    It may also activate co-occurrence topology edges to already-alarming
    connected sites. Matching starts only from the updated alarm edge and those
    active co-occurrence topology edges, avoiding full-graph candidate storage.
    """

    def __init__(
        self,
        topo_downstream_map,
        rules_config,
        site_domain_map,
        alarm_source_domain_map=None,
        site_chain_index=None,
        algorithm="graphflow",
        alarm_active_sec=0.0,
    ):
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"unsupported CSM algorithm: {algorithm}")
        self.algorithm = algorithm
        self.alarm_active_sec = float(alarm_active_sec or 0.0)
        self.rules = {
            name: RulePlan(name, config, algorithm=algorithm)
            for name, config in rules_config.items()
        }
        self.topology = TopologyIndex(topo_downstream_map, site_chain_index=site_chain_index)
        self.active_alarms = ActiveAlarmIndex(alarm_source_domain_map=alarm_source_domain_map)
        self.graph = DynamicFaultGraph(self.topology, self.active_alarms)
        self.predicate = SitePredicate(site_domain_map, self.active_alarms)
        self.role_index = RoleSiteIndex(self.rules, site_domain_map, self.predicate)
        self.node_config_rule_role = {
            id(node_config): (rule_name, role)
            for rule_name, plan in self.rules.items()
            for role, node_config in plan.nodes.items()
        }
        self.global_ttl = 3600.0
        self.power_ttl = 10800.0
        self.watermark = 0.0
        self.history = CSMGroupStore(rules_config, self.global_ttl)
        self.stream_cache = StreamingIntermediateCache(
            self.topology,
            self.role_index,
            algorithm=self.algorithm,
        )

    def process_event(self, item):
        alarm = item["alarm"]
        site_id = item["site_id"]
        alarm_type = item["alarm_title"]
        ts = float(item["ts"])
        eid = alarm.get("告警编码ID", "")
        alarm_source = item.get("alarm_source", "")
        self.watermark = max(self.watermark, ts)
        raw_matches = []
        eval_cache = {}

        expired_events = self.graph.expire_until(ts)
        pruned_events = self.graph.prune_site(site_id, ts, self._event_ttl(alarm_type))
        invalidated_events = [*expired_events, *pruned_events]
        self._handle_invalidated_events(invalidated_events)
        if invalidated_events:
            self._invalidate_stream_dynamic_cache()
            raw_matches.extend(
                self._enumerate_after_alarm_invalidations(
                    invalidated_events,
                    ts,
                    eval_cache,
                )
            )

        if self._is_clear(alarm):
            deleted_event = self.graph.delete_alarm(eid)
            self._handle_invalidated_events([deleted_event])
            if deleted_event:
                self._invalidate_stream_dynamic_cache()
                raw_matches.extend(
                    self._enumerate_after_alarm_invalidations(
                        [deleted_event],
                        ts,
                        eval_cache,
                    )
                )
            return self.history.finalize(raw_matches, self.watermark)

        expire_ts = ts + self.alarm_active_sec if self.alarm_active_sec > 0 else None
        graph_update = self.graph.insert_alarm(
            site_id,
            alarm_type,
            ts,
            eid,
            alarm_source=alarm_source,
            expire_ts=expire_ts,
        )
        updated_alarm_edge = graph_update["alarm_edge"]
        self._invalidate_stream_dynamic_cache()

        raw_matches.extend(
            self._enumerate_from_updated_alarm_edge(
                updated_alarm_edge,
                ts,
                eval_cache,
            )
        )
        return self.history.finalize(raw_matches, self.watermark)

    @staticmethod
    def _is_clear(alarm):
        return str(alarm.get("清除告警", "")).strip().lower() in {"是", "yes", "true", "1", "y"}

    def _event_ttl(self, alarm_type):
        return self.power_ttl if alarm_type in POWER_ALARMS else self.global_ttl

    def _event_matches_expected(self, alarm_type, expected, alarm_source):
        source_domain = self.active_alarms.alarm_source_domain_map.get(alarm_source, "")
        if expected == "ANY":
            return alarm_type in CRITICAL_ALARMS
        if isinstance(expected, dict):
            required_alarms = expected.get("required_alarms")
            if isinstance(required_alarms, Iterable) and not isinstance(required_alarms, str):
                if alarm_type not in required_alarms:
                    return False
                domains = expected.get("required_alarm_source_domains")
                if domains is None:
                    return True
                return any(SitePredicate.has_domain(source_domain, domain) for domain in SitePredicate._normalize_domain_filter(domains))
            optional_alarms = expected.get("optional_alarms")
            if isinstance(optional_alarms, Iterable) and not isinstance(optional_alarms, str):
                return alarm_type in optional_alarms
            return False
        if isinstance(expected, Iterable) and not isinstance(expected, str):
            return alarm_type in expected
        return False

    def flush(self):
        return []

    def _handle_invalidated_events(self, events):
        eids = {
            event.get("eid")
            for event in events
            if event and event.get("eid")
        }
        if eids:
            self.history.invalidate_by_eids(eids)

    def _enumerate_after_alarm_invalidations(self, invalidated_events, ts, eval_cache):
        """Re-match locally after alarm deletion/expiry.

        Deletions can make negative predicates such as NO_OFFLINE become true.
        Instead of scanning the whole active graph, reuse the same CSM add-edge
        path on active alarm edges around the invalidated sites.
        """
        impacted_sites = self._impacted_sites_for_invalidations(invalidated_events)
        if not impacted_sites:
            return []
        matches = []
        seen_active_eids = set()
        for impacted_site in sorted(impacted_sites):
            for active_event in self.active_alarms.by_site.get(impacted_site, ()):
                active_eid = active_event.get("eid")
                if active_eid and active_eid in seen_active_eids:
                    continue
                if active_eid:
                    seen_active_eids.add(active_eid)
                updated_alarm_edge = {
                    "kind": "HAS_ALARM",
                    "src": active_event.get("node"),
                    "dst": active_eid,
                    "site": active_event.get("node"),
                    "alarm": active_event,
                    "label": active_event.get("alarm"),
                }
                matches.extend(
                    self._enumerate_from_updated_alarm_edge(
                        updated_alarm_edge,
                        active_event.get("ts", ts),
                        eval_cache,
                    )
                )
        return matches

    def _impacted_sites_for_invalidations(self, invalidated_events):
        impacted = set()
        for event in invalidated_events:
            if not event:
                continue
            site_id = event.get("node")
            if not site_id:
                continue
            impacted.add(site_id)
            for rule_name, role in self.role_index.event_seed_roles(site_id):
                plan = self.rules.get(rule_name)
                if plan is None:
                    continue
                for edge in plan.pattern_adj.get(role, ()):
                    impacted.update(
                        self._candidate_neighbor_sites(
                            plan,
                            role,
                            site_id,
                            edge["role"],
                            edge,
                        )
                    )
        return impacted

    def _invalidate_stream_dynamic_cache(self):
        self.stream_cache.invalidate_dynamic()

    def _enumerate_from_updated_edge(self, plan, updated_alarm_edge, ts, eval_cache):
        """Enumerate matches containing the inserted HAS_ALARM data edge.

        This mirrors the classic CSM AddEdge path: the data update is the real
        edge ``site --HAS_ALARM--> alarm``. A topology query edge can only start
        enumeration if the corresponding site-site edge is active because both
        endpoints currently have co-occurring alarms.
        """
        if updated_alarm_edge.get("kind") != "HAS_ALARM":
            return []
        updated_site = updated_alarm_edge["site"]
        alarm_event = updated_alarm_edge["alarm"]
        alarm_type = alarm_event.get("alarm")
        alarm_source = alarm_event.get("alarm_source", "")
        matches = []
        for updated_role in self._event_seed_roles_for_plan(plan, updated_site):
            alarm_query_edge = plan.alarm_query_edges[updated_role]
            node_config = alarm_query_edge["node_config"]
            if not self._event_can_seed_role(updated_site, node_config, alarm_type, alarm_source):
                continue
            valid, updated_events = self._validate_cached(
                eval_cache,
                updated_site,
                updated_role,
                node_config,
                ts,
                0,
            )
            if not valid or not self._events_contain_alarm(updated_events, alarm_type, alarm_source, ts):
                continue

            if not plan.pattern_adj.get(updated_role):
                match = self._build_match_from_mapping(
                    plan,
                    {updated_role: (updated_site, updated_events)},
                    ts,
                )
                if match is not None and self._validate_result_constraints(plan.config, match):
                    matches.append(match)
                continue

            for edge in plan.pattern_adj.get(updated_role, []):
                neighbor_role = edge["role"]
                neighbor_config = plan.nodes[neighbor_role]
                for neighbor_site in self._seed_neighbors(
                    updated_site,
                    edge,
                    plan=plan,
                    source_role=updated_role,
                    target_role=neighbor_role,
                ):
                    if edge.get("dedupe_symmetric_pair") and str(updated_site) > str(neighbor_site):
                        continue
                    valid_neighbor, neighbor_events = self._validate_cached(
                        eval_cache,
                        neighbor_site,
                        neighbor_role,
                        neighbor_config,
                        ts,
                        edge.get("win", 0),
                    )
                    if not valid_neighbor:
                        continue
                    if not self._algorithm_candidate_supported(plan, updated_role, updated_site, ts, eval_cache):
                        continue
                    if not self._algorithm_candidate_supported(plan, neighbor_role, neighbor_site, ts, eval_cache):
                        continue
                    if self.algorithm == "turboflux":
                        matches.extend(
                            self._turboflux_find_matches(
                                plan,
                                updated_role,
                                updated_site,
                                updated_events,
                                neighbor_role,
                                neighbor_site,
                                neighbor_events,
                                ts,
                                eval_cache,
                            )
                        )
                    else:
                        matches.extend(
                            self._graphflow_find_matches(
                                plan,
                                updated_role,
                                updated_site,
                                updated_events,
                                neighbor_role,
                                neighbor_site,
                                neighbor_events,
                                ts,
                                eval_cache,
                            )
                        )
        return matches

    def _seed_neighbors(self, updated_site, edge, plan=None, source_role=None, target_role=None):
        if (
            self.stream_cache is not None
            and plan is not None
            and source_role is not None
            and target_role is not None
        ):
            yield from self.stream_cache.candidate_neighbors(
                plan,
                source_role,
                updated_site,
                target_role,
                edge,
            )
            return
        for candidate_site, _hop in self.graph.reachable_sites(updated_site, edge):
            yield candidate_site

    def _event_can_seed_role(self, site_id, node_config, alarm_type, alarm_source):
        if not self.predicate.matches_structure(site_id, node_config):
            return False
        if node_config.get("type", "primitive") == "compound":
            return any(
                self._event_can_seed_role(site_id, pattern, alarm_type, alarm_source)
                for pattern in node_config.get("patterns", [])
            )
        expected = self.predicate.resolve_expected_alarms(site_id, node_config)
        return self._event_matches_expected(alarm_type, expected, alarm_source)

    @staticmethod
    def _events_contain_alarm(events, alarm_type, alarm_source, ts):
        for event in events:
            if event.get("alarm") != alarm_type:
                continue
            if alarm_source and event.get("alarm_source") != alarm_source:
                continue
            if event.get("ts") == ts:
                return True
        return False

    def _graphflow_find_matches(
        self,
        plan,
        first_role,
        first_site,
        first_events,
        second_role,
        second_site,
        second_events,
        trigger_ts,
        eval_cache,
    ):
        order_info = plan.query_orders.get((first_role, second_role))
        if order_info is None:
            return []

        mapping = {
            first_role: (first_site, first_events),
            second_role: (second_site, second_events),
        }
        visited_sites = {first_site, second_site}
        results = []
        self._graphflow_backtrack(
            plan,
            order_info,
            depth=2,
            mapping=mapping,
            visited_sites=visited_sites,
            trigger_ts=trigger_ts,
            eval_cache=eval_cache,
            results=results,
        )
        return results

    def _turboflux_find_matches(
        self,
        plan,
        first_role,
        first_site,
        first_events,
        second_role,
        second_site,
        second_events,
        trigger_ts,
        eval_cache,
    ):
        """TurboFlux matching using DCS index and join_check arrays."""
        # Rule semantics stay identical to the common backtracker.  The
        # TurboFlux-specific part is in query ordering plus cached local
        # candidate/support intermediates used by _graphflow_candidate_sites()
        # and _dcs_candidate_supported().
        return self._graphflow_find_matches(
            plan,
            first_role,
            first_site,
            first_events,
            second_role,
            second_site,
            second_events,
            trigger_ts,
            eval_cache,
        )

    def _enumerate_negative_matches(self, rule_name, plan, deleted_event, eval_cache):
        """Enumerate matches destroyed by an alarm deletion (negative matching).

        In the current rule-compatible CSM path, deletion mainly invalidates
        already emitted groups through their eids.  This helper is kept for
        callers that want to inspect the matches around the deleted HAS_ALARM
        edge before dynamic caches are invalidated.
        """
        if deleted_event is None:
            return []

        deleted_site = deleted_event.get("node")
        alarm_type = deleted_event.get("alarm")
        alarm_source = deleted_event.get("alarm_source", "")
        deleted_ts = deleted_event.get("ts", 0)
        matches = []

        for role, alarm_query_edge in plan.alarm_query_edges.items():
            node_config = alarm_query_edge["node_config"]
            if not self._event_can_seed_role(deleted_site, node_config, alarm_type, alarm_source):
                continue
            valid, events = self._validate_cached(
                eval_cache, deleted_site, role, node_config, deleted_ts, 0
            )
            if not valid:
                continue
            if not plan.pattern_adj.get(role):
                match = self._build_match_from_mapping(
                    plan, {role: (deleted_site, events)}, deleted_ts
                )
                if match is not None and self._validate_result_constraints(plan.config, match):
                    matches.append(match)
                continue

            for edge in plan.pattern_adj.get(role, []):
                neighbor_role = edge["role"]
                neighbor_config = plan.nodes[neighbor_role]
                for neighbor_site in self._seed_neighbors(
                    deleted_site,
                    edge,
                    plan=plan,
                    source_role=role,
                    target_role=neighbor_role,
                ):
                    if edge.get("dedupe_symmetric_pair") and str(deleted_site) > str(neighbor_site):
                        continue
                    valid_n, n_events = self._validate_cached(
                        eval_cache, neighbor_site, neighbor_role, neighbor_config,
                        deleted_ts, edge.get("win", 0)
                    )
                    if not valid_n:
                        continue
                    matches.extend(
                        self._turboflux_find_matches(
                            plan, role, deleted_site, events,
                            neighbor_role, neighbor_site, n_events,
                            deleted_ts, eval_cache,
                        )
                    )
        return matches

    def _enumerate_from_updated_alarm_edge(self, updated_alarm_edge, ts, eval_cache):
        if updated_alarm_edge.get("kind") != "HAS_ALARM":
            return []
        updated_site = updated_alarm_edge["site"]
        matches = []
        seen_plans = set()
        for rule_name, role in self.role_index.event_seed_roles(updated_site):
            plan = self.rules.get(rule_name)
            if plan is None or role not in plan.alarm_query_edges or rule_name in seen_plans:
                continue
            seen_plans.add(rule_name)
            matches.extend(
                self._enumerate_from_updated_edge(
                    plan,
                    updated_alarm_edge,
                    ts,
                    eval_cache,
                )
            )
        return matches

    def _event_seed_roles_for_plan(self, plan, updated_site):
        return tuple(
            role
            for rule_name, role in self.role_index.event_seed_roles(updated_site)
            if rule_name == plan.name and role in plan.alarm_query_edges
        )

    def _graphflow_backtrack(self, plan, order_info, depth, mapping, visited_sites, trigger_ts, eval_cache, results):
        order = order_info["order"]
        if depth >= len(order):
            match = self._build_match_from_mapping(plan, mapping, trigger_ts)
            if match is not None and self._validate_result_constraints(plan.config, match):
                results.append(match)
            return

        role = order[depth]
        candidate_sites = self._graphflow_candidate_sites(plan, order_info, role, mapping, eval_cache)
        for site_id, events in candidate_sites:
            if site_id in visited_sites:
                continue
            mapping[role] = (site_id, events)
            visited_sites.add(site_id)
            self._graphflow_backtrack(
                plan,
                order_info,
                depth + 1,
                mapping,
                visited_sites,
                trigger_ts,
                eval_cache,
                results,
            )
            visited_sites.remove(site_id)
            mapping.pop(role, None)

    def _graphflow_candidate_sites(self, plan, order_info, role, mapping, eval_cache):
        backward_roles = [item for item in order_info["backward"].get(role, ()) if item in mapping]
        if not backward_roles:
            return []

        best_role = None
        best_edge = None
        best_candidates = None
        for matched_role in backward_roles:
            edge = self._query_edge(plan, matched_role, role)
            if edge is None:
                continue
            matched_site, _matched_events = mapping[matched_role]
            candidates = {
                site_id: 1
                for site_id in self._candidate_neighbor_sites(
                    plan,
                    matched_role,
                    matched_site,
                    role,
                    edge,
                )
            }
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_role = matched_role
                best_edge = edge
                best_candidates = candidates
        if best_candidates is None:
            return []

        result = []
        node_config = plan.nodes[role]
        for candidate_site in best_candidates:
            all_events = []
            joinable = True
            for matched_role in backward_roles:
                matched_site, matched_events = mapping[matched_role]
                edge = self._query_edge(plan, matched_role, role)
                if edge is None:
                    joinable = False
                    break
                if not self.graph.has_topology_relation(
                    matched_site,
                    candidate_site,
                    edge,
                ):
                    joinable = False
                    break
                ref_ts = matched_events[0]["ts"] if matched_events else mapping[best_role][1][0]["ts"] if mapping[best_role][1] else 0
                valid, events = self._validate_cached(
                    eval_cache,
                    candidate_site,
                    role,
                    node_config,
                    ref_ts,
                    edge.get("win", 0),
                )
                if not valid:
                    joinable = False
                    break
                all_events.extend(events)
            if joinable and self._algorithm_candidate_supported(
                plan,
                role,
                candidate_site,
                trigger_ts=all_events[0]["ts"] if all_events else 0,
                eval_cache=eval_cache,
            ):
                result.append((candidate_site, SitePredicate._dedupe_events(all_events)))
        if self.algorithm in {"iedyn", "symbi", "turboflux"}:
            result.sort(
                key=lambda item: (
                    self._candidate_support_count(plan, role, item[0], item[1][0]["ts"] if item[1] else 0, eval_cache),
                    plan._role_selectivity_rank(role),
                    item[0],
                )
            )
        elif self.algorithm == "sjtree":
            result.sort(key=lambda item: (len(item[1]), item[0]))
        return result

    def _algorithm_candidate_supported(self, plan, role, site_id, trigger_ts, eval_cache):
        if self.algorithm in {"incisomatch", "sjtree", "graphflow"}:
            return True
        return self._dcs_candidate_supported(plan, role, site_id, trigger_ts, eval_cache)

    def _dcs_candidate_supported(self, plan, role, site_id, trigger_ts, eval_cache):
        """SymBi/TurboFlux-style candidate support check.

        This is the Python data-model equivalent of DCS feasibility: every
        query neighbor that is not yet bound must have at least one active data
        neighbor satisfying the neighbor role. The actual full joinability is
        still verified by backtracking.
        """
        cache_key = ("dcs", plan.name, role, site_id, trigger_ts)
        if cache_key in eval_cache:
            return eval_cache[cache_key]
        if self.stream_cache.supports_dynamic_pruning:
            cached = self.stream_cache.support_get(cache_key)
            if cached is not None:
                eval_cache[cache_key] = cached
                return cached
        for edge in plan.pattern_adj.get(role, ()):
            neighbor_role = edge["role"]
            neighbor_config = plan.nodes[neighbor_role]
            has_support = False
            for neighbor_site in self._candidate_neighbor_sites(
                plan,
                role,
                site_id,
                neighbor_role,
                edge,
            ):
                valid, _events = self._validate_cached(
                    eval_cache,
                    neighbor_site,
                    neighbor_role,
                    neighbor_config,
                    trigger_ts,
                    edge.get("win", 0),
                )
                if valid:
                    has_support = True
                    break
            if not has_support and not edge.get("optional"):
                eval_cache[cache_key] = False
                if self.stream_cache.supports_dynamic_pruning:
                    self.stream_cache.support_set(cache_key, False)
                return False
        eval_cache[cache_key] = True
        if self.stream_cache.supports_dynamic_pruning:
            self.stream_cache.support_set(cache_key, True)
        return True

    def _candidate_support_count(self, plan, role, site_id, trigger_ts, eval_cache):
        cache_key = ("support_count", plan.name, role, site_id, trigger_ts)
        if cache_key in eval_cache:
            return eval_cache[cache_key]
        if self.stream_cache.supports_dynamic_pruning:
            cached = self.stream_cache.support_get(cache_key)
            if cached is not None:
                eval_cache[cache_key] = cached
                return cached
        total = 0
        for edge in plan.pattern_adj.get(role, ()):
            neighbor_role = edge["role"]
            neighbor_config = plan.nodes[neighbor_role]
            for neighbor_site in self._candidate_neighbor_sites(
                plan,
                role,
                site_id,
                neighbor_role,
                edge,
            ):
                valid, _events = self._validate_cached(
                    eval_cache,
                    neighbor_site,
                    neighbor_role,
                    neighbor_config,
                    trigger_ts,
                    edge.get("win", 0),
                )
                if valid:
                    total += 1
        eval_cache[cache_key] = total
        if self.stream_cache.supports_dynamic_pruning:
            self.stream_cache.support_set(cache_key, total)
        return total

    def _candidate_neighbor_sites(self, plan, source_role, source_site, target_role, edge):
        return self.stream_cache.candidate_neighbors(
            plan,
            source_role,
            source_site,
            target_role,
            edge,
        )

    @staticmethod
    def _query_edge(plan, source_role, target_role):
        for edge in plan.pattern_adj.get(source_role, []):
            if edge["role"] == target_role:
                return edge
        return None

    def _build_match_from_mapping(self, plan, mapping, trigger_ts):
        inst = {
            "roles": {
                role: {"nodes": {site_id: events}, "checked": True}
                for role, (site_id, events) in mapping.items()
            }
        }
        return self._build_match(plan, inst, trigger_ts)

    def _validate_cached(self, eval_cache, site_id, role, node_config, reference_ts, edge_window):
        # Fast static role/site pruning. Dynamic alarm predicates are still
        # evaluated below so context/no-alarm roles keep fault_grouping semantics.
        rule_role = self.node_config_rule_role.get(id(node_config))
        if rule_role and not self.role_index.matches(rule_role[0], role, site_id):
            return False, []
        cache_key = (site_id, role, id(node_config), reference_ts, self._window_key(edge_window))
        if cache_key not in eval_cache:
            eval_cache[cache_key] = self.predicate.validate(site_id, node_config, reference_ts, edge_window)
        return eval_cache[cache_key]

    @staticmethod
    def _window_key(edge_window):
        if isinstance(edge_window, dict):
            return tuple(sorted(edge_window.items()))
        return edge_window

    def _build_match(self, plan, inst, trigger_ts):
        role_mapping = {}
        symptoms = {}
        for role, state in inst.get("roles", {}).items():
            nodes = sorted(site_id for site_id in state.get("nodes", {}) if site_id)
            if nodes:
                role_mapping[role] = nodes
            for site_id, events in state.get("nodes", {}).items():
                for event in events:
                    eid = event.get("eid")
                    if not eid:
                        continue
                    symptom = dict(event)
                    symptom["matched_role"] = role
                    symptom["time_str"] = datetime.fromtimestamp(event["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                    symptoms[eid] = symptom
        if not symptoms:
            return None
        inferred_roots = {
            role: role_mapping.get(role, [])
            for role in plan.root_roles
            if role_mapping.get(role)
        }
        return {
            "uuid": str(uuid.uuid4()),
            "rule": plan.name,
            "merged_rules": [plan.name],
            "inferred_roots": inferred_roots,
            "role_mapping": role_mapping,
            "symptoms": sorted(symptoms.values(), key=lambda item: (item.get("ts", 0), item.get("eid", ""))),
            "_expire_ts_hint": min(event["ts"] for event in symptoms.values()) + plan.config.get("max_stay_time_sec", self.global_ttl),
        }

    @staticmethod
    def _validate_result_constraints(rule, match):
        constraints = rule.get("result_constraints") or {}
        for item in constraints.get("role_alarm_or_presence_any", []):
            min_matches = int(item.get("min_matches", 1))
            matched = 0
            alarm_roles = set(item.get("alarm_roles") or [])
            alarms = set(item.get("alarms") or [])
            presence_roles = set(item.get("presence_roles") or [])
            for symptom in match.get("symptoms", []):
                if symptom.get("matched_role") in alarm_roles and symptom.get("alarm") in alarms:
                    matched += 1
            for role in presence_roles:
                matched += len(match.get("role_mapping", {}).get(role, []))
            if matched < min_matches:
                return False
        return True


class CSMGroupStore:
    def __init__(self, rules_config, default_ttl):
        self.rules = rules_config
        self.default_ttl = default_ttl
        self.groups = []
        self.eid_to_group = collections.defaultdict(set)

    def finalize(self, matches, current_ts):
        self.prune(current_ts)
        output = []
        for match in matches:
            merged, related_indexes, should_emit = self.merge(match)
            if should_emit:
                output.append(merged)
            self.store(merged, related_indexes)
        return output

    def prune(self, current_ts):
        kept = []
        for item in self.groups:
            if item is not None and current_ts <= item["expire_ts"]:
                kept.append(item)
        if len(kept) != len(self.groups):
            self.groups = kept
            self._rebuild_index()

    def invalidate_by_eids(self, eids):
        related_indexes = {
            idx
            for eid in eids
            for idx in self.eid_to_group.get(eid, ())
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None
        }
        if not related_indexes:
            return 0
        for idx in related_indexes:
            self.groups[idx] = None
        self._rebuild_index()
        return len(related_indexes)

    def merge(self, match):
        eids = self._match_eids(match)
        if not eids:
            return match, set(), True
        related_indexes = {
            idx
            for eid in eids
            for idx in self.eid_to_group.get(eid, ())
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None
        }
        if not related_indexes:
            return match, set(), True

        merged = {
            "uuid": match.get("uuid"),
            "rule": match.get("rule"),
            "merged_rules": set(match.get("merged_rules", [match.get("rule")])),
            "inferred_roots": copy.deepcopy(match.get("inferred_roots", {})),
            "role_mapping": copy.deepcopy(match.get("role_mapping", {})),
            "symptoms": {symptom.get("eid"): symptom for symptom in match.get("symptoms", []) if symptom.get("eid")},
            "_expire_ts_hint": match.get("_expire_ts_hint"),
        }
        fully_contained = False
        for idx in related_indexes:
            previous = self.groups[idx]["match"]
            previous_eids = self._match_eids(previous)
            if eids.issubset(previous_eids):
                fully_contained = True
            merged["merged_rules"].update(previous.get("merged_rules", [previous.get("rule")]))
            for role, nodes in previous.get("role_mapping", {}).items():
                merged["role_mapping"].setdefault(role, [])
                merged["role_mapping"][role] = sorted(set(merged["role_mapping"][role]) | set(nodes))
            for role, nodes in previous.get("inferred_roots", {}).items():
                merged["inferred_roots"].setdefault(role, [])
                merged["inferred_roots"][role] = sorted(set(merged["inferred_roots"][role]) | set(nodes))
            for symptom in previous.get("symptoms", []):
                if symptom.get("eid"):
                    merged["symptoms"][symptom["eid"]] = symptom

        result = dict(merged)
        result["merged_rules"] = sorted(rule for rule in result["merged_rules"] if rule)
        result["symptoms"] = sorted(result["symptoms"].values(), key=lambda item: (item.get("ts", 0), item.get("eid", "")))
        return result, related_indexes, not fully_contained

    def store(self, match, related_indexes):
        for idx in related_indexes:
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None:
                self.groups[idx] = None
        anchor_ts = min((symptom["ts"] for symptom in match.get("symptoms", []) if "ts" in symptom), default=0)
        expire_ts = match.pop("_expire_ts_hint", None)
        if expire_ts is None:
            expire_ts = anchor_ts + self.rules.get(match.get("rule"), {}).get("max_stay_time_sec", self.default_ttl)
        item = {"match": match, "expire_ts": expire_ts}
        self.groups.append(item)
        idx = len(self.groups) - 1
        for eid in self._match_eids(match):
            self.eid_to_group[eid].add(idx)
        if len(related_indexes) > 64:
            self._rebuild_index()

    def _rebuild_index(self):
        self.eid_to_group.clear()
        for idx, item in enumerate(self.groups):
            if item is None:
                continue
            for eid in self._match_eids(item["match"]):
                self.eid_to_group[eid].add(idx)

    @staticmethod
    def _match_eids(match):
        return {
            symptom.get("eid")
            for symptom in match.get("symptoms", [])
            if symptom.get("eid")
        }
