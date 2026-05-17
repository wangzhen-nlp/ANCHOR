import collections
import json
import math
import threading


class BatchSiteMergeHelper:
    """统一管理批内站点级弱合并策略。

    当前支持两类策略：
    1. 基于站点拓扑 hop 邻接的合并
    2. 基于站点局部密度的自适应空间合并

    这样 TemporalGraphEngine 只需要知道“是否可合并”，
    不再关心 hop 缓存、站点邻接搜索或空间判定细节。
    """

    def __init__(
        self,
        topo_downstream_map,
        site_neighbor_hops=0,
        density_helper=None,
        max_site_hop_cache_size=20000,
    ):
        self.site_neighbor_hops = max(int(site_neighbor_hops or 0), 0)
        self.density_helper = density_helper
        self.max_site_hop_cache_size = max(int(max_site_hop_cache_size or 0), 1)

        self._lock = threading.RLock()
        self._site_hop_cache = collections.OrderedDict()
        self._topo_undirected = collections.defaultdict(set)

        for up, downs in topo_downstream_map.items():
            for down in downs:
                self._topo_undirected[up].add(down)
                self._topo_undirected[down].add(up)

    @property
    def enabled(self):
        return (
            self.site_neighbor_hops > 0
            or (self.density_helper is not None and self.density_helper.enabled)
        )

    def warmup(self):
        if self.density_helper is not None and self.density_helper.enabled:
            self.density_helper.warmup()

    def are_components_adjacent(self, left_sites, right_sites):
        return self.classify_component_adjacency(left_sites, right_sites) is not None

    def classify_component_adjacency(self, left_sites, right_sites):
        if not left_sites or not right_sites:
            return None
        if set(left_sites) & set(right_sites):
            return "shared_site"

        if self.site_neighbor_hops > 0 and self._are_sites_hop_adjacent(left_sites, right_sites):
            return "hop"

        if self.density_helper is not None and self.density_helper.enabled:
            for left_site in left_sites:
                for right_site in right_sites:
                    if self.density_helper.can_merge_site_pair(left_site, right_site):
                        return "distance"

        return None

    def _are_sites_hop_adjacent(self, left_sites, right_sites):
        smaller_sites, larger_sites = (left_sites, right_sites)
        if len(smaller_sites) > len(larger_sites):
            smaller_sites, larger_sites = larger_sites, smaller_sites

        larger_sites = set(larger_sites)
        for site in smaller_sites:
            if self._get_sites_within_hops(site, self.site_neighbor_hops) & larger_sites:
                return True
        return False

    def _get_sites_within_hops(self, start_site, max_hops):
        max_hops = max(int(max_hops or 0), 0)
        cache_key = (start_site, max_hops)

        with self._lock:
            if cache_key in self._site_hop_cache:
                self._site_hop_cache.move_to_end(cache_key)
                return self._site_hop_cache[cache_key]

        visited = {start_site}
        reachable = {start_site}
        queue = collections.deque([(start_site, 0)])

        while queue:
            curr_site, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for next_site in self._topo_undirected.get(curr_site, ()):
                if next_site in visited:
                    continue
                visited.add(next_site)
                reachable.add(next_site)
                queue.append((next_site, hops + 1))

        reachable = frozenset(reachable)
        with self._lock:
            self._site_hop_cache[cache_key] = reachable
            self._site_hop_cache.move_to_end(cache_key)
            if len(self._site_hop_cache) > self.max_site_hop_cache_size:
                self._site_hop_cache.popitem(last=False)

        return reachable


class AdaptiveDensitySiteMergeHelper:
    """基于站点局部密度的自适应空间合并辅助器。

    设计目标：
    1. 不把空间合并逻辑继续堆进 match_rules.py / temporal_graph_engine.py。
    2. 利用 KD-tree 加速“局部密度半径”的近邻搜索，但默认路径下不强依赖 numpy/scipy。
    3. 合并阈值不使用全局固定距离，而是按站点自己的局部密度自适应放缩。

    判定方式：
    - 先取站点的第 k 近邻距离作为局部密度尺度。
    - 再乘上 density_scale 得到该站点的自适应半径。
    - 两个站点之间若真实球面距离 <= max(左站半径, 右站半径)，则认为可做批内弱合并。
    """

    EARTH_RADIUS_M = 6371000.0
    _CACHE_MISS = object()

    def __init__(
        self,
        site_graph_path,
        density_knn,
        density_scale=1.0,
        min_radius_meters=0.0,
        max_radius_meters=0.0,
        leafsize=10,
        max_radius_cache_size=50000,
        max_pair_cache_size=200000,
    ):
        self.site_graph_path = site_graph_path
        self.density_knn = max(int(density_knn or 0), 0)
        self.density_scale = max(float(density_scale or 0.0), 0.0)
        self.min_radius_meters = max(float(min_radius_meters or 0.0), 0.0)
        self.max_radius_meters = max(float(max_radius_meters or 0.0), 0.0)
        self.leafsize = max(int(leafsize or 10), 1)

        self.max_radius_cache_size = max(int(max_radius_cache_size or 0), 1)
        self.max_pair_cache_size = max(int(max_pair_cache_size or 0), 1)

        self._lock = threading.RLock()
        self._geo_tree = None
        self._site_coords = self._load_site_coords(site_graph_path)
        self._radius_cache = collections.OrderedDict()
        self._distance_cache = collections.OrderedDict()
        self._pair_merge_cache = collections.OrderedDict()

    @property
    def enabled(self):
        return self.density_knn > 0 and self.density_scale > 0

    def warmup(self):
        """显式预热 KD-tree，便于在脚本启动阶段就尽早暴露依赖问题。"""
        if not self.enabled:
            return
        self._ensure_geo_tree()

    def can_merge_site_pair(self, left_site, right_site):
        """判断两个站点是否满足基于局部密度的弱空间合并条件。"""
        if not self.enabled:
            return False

        left_site = str(left_site or "").strip()
        right_site = str(right_site or "").strip()
        if not left_site or not right_site:
            return False
        if left_site == right_site:
            return True

        cache_key = self._normalize_pair_key(left_site, right_site)
        cached_value = self._cache_get(self._pair_merge_cache, cache_key)
        if cached_value is not self._CACHE_MISS:
            return cached_value

        left_radius = self.get_site_density_radius(left_site)
        right_radius = self.get_site_density_radius(right_site)
        if left_radius <= 0 and right_radius <= 0:
            self._cache_put(self._pair_merge_cache, cache_key, False, self.max_pair_cache_size)
            return False

        actual_distance = self.get_site_pair_distance(left_site, right_site)
        if actual_distance is None:
            self._cache_put(self._pair_merge_cache, cache_key, False, self.max_pair_cache_size)
            return False

        adaptive_threshold = max(left_radius, right_radius)
        can_merge = actual_distance <= adaptive_threshold
        self._cache_put(self._pair_merge_cache, cache_key, can_merge, self.max_pair_cache_size)
        return can_merge

    def get_site_density_radius(self, site_id):
        """返回单站点的自适应空间半径。"""
        if not self.enabled:
            return 0.0

        site_id = str(site_id or "").strip()
        if not site_id:
            return 0.0

        cached_value = self._cache_get(self._radius_cache, site_id)
        if cached_value is not self._CACHE_MISS:
            return cached_value

        if site_id not in self._site_coords:
            self._cache_put(self._radius_cache, site_id, 0.0, self.max_radius_cache_size)
            return 0.0

        effective_k = min(self.density_knn, max(len(self._site_coords) - 1, 0))
        if effective_k <= 0:
            self._cache_put(self._radius_cache, site_id, 0.0, self.max_radius_cache_size)
            return 0.0

        neighbors = self._ensure_geo_tree().nearest_neighbors(site_id, k=effective_k)
        neighbor_distances = [
            float(item.get("distance", 0.0))
            for item in neighbors
            if item.get("id") and item.get("distance") is not None
        ]

        raw_radius = max(neighbor_distances) if neighbor_distances else 0.0
        adaptive_radius = raw_radius * self.density_scale

        if self.min_radius_meters > 0:
            adaptive_radius = max(adaptive_radius, self.min_radius_meters)
        if self.max_radius_meters > 0:
            adaptive_radius = min(adaptive_radius, self.max_radius_meters)

        adaptive_radius = max(adaptive_radius, 0.0)
        self._cache_put(self._radius_cache, site_id, adaptive_radius, self.max_radius_cache_size)
        return adaptive_radius

    def get_site_pair_distance(self, left_site, right_site):
        left_site = str(left_site or "").strip()
        right_site = str(right_site or "").strip()
        if not left_site or not right_site:
            return None
        if left_site == right_site:
            return 0.0

        cache_key = self._normalize_pair_key(left_site, right_site)
        cached_value = self._cache_get(self._distance_cache, cache_key)
        if cached_value is not self._CACHE_MISS:
            return cached_value

        left_coord = self._site_coords.get(left_site)
        right_coord = self._site_coords.get(right_site)
        if left_coord is None or right_coord is None:
            self._cache_put(self._distance_cache, cache_key, None, self.max_pair_cache_size)
            return None

        distance_m = self._haversine_distance(
            left_coord[0],
            left_coord[1],
            right_coord[0],
            right_coord[1],
        )
        self._cache_put(self._distance_cache, cache_key, distance_m, self.max_pair_cache_size)
        return distance_m

    def _ensure_geo_tree(self):
        if self._geo_tree is not None:
            return self._geo_tree

        with self._lock:
            if self._geo_tree is not None:
                return self._geo_tree

            try:
                from topology_tools.geokdtree import GeoKDTree
            except Exception as exc:
                raise RuntimeError(
                    "启用基于密度的站点批内合并时，加载 topology_tools.geokdtree 失败。"
                    "请确认 numpy/scipy 可用。"
                ) from exc

            self._geo_tree = GeoKDTree(leafsize=self.leafsize).build(self.site_graph_path)
            return self._geo_tree

    @staticmethod
    def _normalize_pair_key(left_site, right_site):
        return (left_site, right_site) if left_site <= right_site else (right_site, left_site)

    def _cache_get(self, cache, key):
        with self._lock:
            if key not in cache:
                return self._CACHE_MISS
            cache.move_to_end(key)
            return cache[key]

    def _cache_put(self, cache, key, value, max_size):
        with self._lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > max_size:
                cache.popitem(last=False)

    @classmethod
    def _haversine_distance(cls, lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        half_chord = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        angular_distance = 2 * math.asin(min(1.0, math.sqrt(max(0.0, half_chord))))
        return cls.EARTH_RADIUS_M * angular_distance

    @staticmethod
    def _load_site_coords(site_graph_path):
        with open(site_graph_path, "r", encoding="utf-8") as file_obj:
            site_graph = json.load(file_obj)

        site_coords = {}
        for site_id, site_info in site_graph.items():
            longitude = str(site_info.get("longitude", "")).strip()
            latitude = str(site_info.get("latitude", "")).strip()
            if not longitude or not latitude:
                continue
            try:
                site_coords[str(site_id)] = (float(latitude), float(longitude))
            except (TypeError, ValueError):
                continue
        return site_coords
