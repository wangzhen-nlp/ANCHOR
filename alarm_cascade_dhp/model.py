import copy
import math
import random

from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from alarm_cascade_dhp.config import AlarmDHPConfig
from alarm_cascade_dhp.topology import TopologyIndex
from alarm_cascade_dhp.event_types import CascadeDecision


@dataclass
class _SupportAlarm:
    event_id: str
    event_key: str
    ts: float
    alarm_title: str
    alarm_source: str
    site_id: str
    device_domain: str

    @classmethod
    def from_event(cls, event):
        return cls(
            event_id=event.event_id,
            event_key=event.event_key,
            ts=event.ts,
            alarm_title=event.alarm_title,
            alarm_source=event.alarm_source,
            site_id=event.site_id,
            device_domain=event.device_domain,
        )


@dataclass
class _Proposal:
    cluster_id: str
    probability: float
    log_score: float
    time_rate: float
    topology_affinity: float
    content_log_score: float


@dataclass
class _Cluster:
    cascade_id: str
    created_ts: float
    last_ts: float
    kernel_alpha: list
    feature_counts: Counter = field(default_factory=Counter)
    total_feature_count: int = 0
    supports: list = field(default_factory=list)
    event_ids: list = field(default_factory=list)
    alarm_titles: Counter = field(default_factory=Counter)
    sites: Counter = field(default_factory=Counter)
    alarm_sources: Counter = field(default_factory=Counter)
    active_event_keys: Counter = field(default_factory=Counter)
    topology_relation_counts: Counter = field(default_factory=Counter)

    def state_at(self, now_ts, config):
        age = max(0.0, now_ts - self.last_ts)
        if age >= config.close_after_sec:
            return "closed"
        if age >= config.cooling_after_sec or not self.active_event_keys:
            return "cooling"
        return "active"

    def is_candidate(self, event_ts, config):
        age = event_ts - self.last_ts
        return 0 <= age <= config.active_window_sec and self.state_at(event_ts, config) != "closed"

    def kernel_weights(self):
        total = sum(self.kernel_alpha)
        if total <= 0:
            return [1.0 / len(self.kernel_alpha)] * len(self.kernel_alpha)
        return [value / total for value in self.kernel_alpha]

    def time_rate(self, event_ts, config):
        weights = self.kernel_weights()
        rate = 0.0
        for support in reversed(self.supports):
            dt = event_ts - support.ts
            if dt < 0:
                continue
            if dt > config.active_window_sec:
                break
            rate += _mixture_rbf(
                dt,
                weights,
                config.time_kernel_means_sec,
                config.time_kernel_bandwidths_sec,
            )
        return rate

    def content_log_predictive(self, feature_counts, vocabulary_size, theta0):
        doc_count = sum(feature_counts.values())
        if doc_count <= 0:
            return 0.0
        vocabulary_size = max(int(vocabulary_size), 1)
        prior_sum = vocabulary_size * theta0
        score = math.lgamma(self.total_feature_count + prior_sum)
        score -= math.lgamma(self.total_feature_count + doc_count + prior_sum)
        for token, count in feature_counts.items():
            old_count = self.feature_counts.get(token, 0)
            score += math.lgamma(old_count + count + theta0)
            score -= math.lgamma(old_count + theta0)
        return score

    def topology_affinity(self, event, topology, config):
        if not self.supports:
            return 1.0
        affinity = 0.0
        saw_relation = False
        for support in reversed(self.supports[-config.max_support_events :]):
            relation = topology.relation(event, support, max_hops=config.topology_max_hops)
            base_weight = config.topology_relation_weights.get(
                relation,
                config.topology_relation_weights["unknown"],
            )
            relation_count = self.topology_relation_counts.get(relation, 0)
            learned_weight = (
                config.topology_prior_mass * base_weight + relation_count
            ) / (config.topology_prior_mass + relation_count)
            affinity = max(affinity, learned_weight)
            saw_relation = True
            if affinity >= 1.0:
                break
        return affinity if saw_relation else 1.0

    def add(self, event, topology, config):
        old_supports = self.supports[-config.max_support_events :]
        self._update_time_kernel(event, old_supports, config)
        self._update_topology_counts(event, old_supports, topology, config)
        self.feature_counts.update(event.feature_counts)
        self.total_feature_count += sum(event.feature_counts.values())
        self.event_ids.append(event.event_id)
        if event.alarm_title:
            self.alarm_titles[event.alarm_title] += 1
        if event.site_id:
            self.sites[event.site_id] += 1
        if event.alarm_source:
            self.alarm_sources[event.alarm_source] += 1
        if event.event_key:
            self.active_event_keys[event.event_key] += 1
        self.supports.append(_SupportAlarm.from_event(event))
        if len(self.supports) > config.max_support_events:
            self.supports = self.supports[-config.max_support_events :]
        self.last_ts = max(self.last_ts, event.ts)

    def clear_event_key(self, event_key):
        if not event_key or event_key not in self.active_event_keys:
            return False
        self.active_event_keys[event_key] -= 1
        if self.active_event_keys[event_key] <= 0:
            del self.active_event_keys[event_key]
        return True

    def snapshot(self, now_ts, config):
        return {
            "cascade_id": self.cascade_id,
            "state": self.state_at(now_ts, config),
            "created_ts": self.created_ts,
            "last_ts": self.last_ts,
            "event_count": len(self.event_ids),
            "event_ids": list(self.event_ids),
            "active_alarm_key_count": sum(self.active_event_keys.values()),
            "sites": [name for name, _ in self.sites.most_common()],
            "alarm_sources": [name for name, _ in self.alarm_sources.most_common()],
            "alarm_titles": [name for name, _ in self.alarm_titles.most_common()],
            "top_features": [
                {"token": token, "count": count}
                for token, count in self.feature_counts.most_common(20)
            ],
            "time_kernel_weights": self.kernel_weights(),
            "topology_relation_counts": dict(self.topology_relation_counts),
        }

    def _update_time_kernel(self, event, old_supports, config):
        weights = self.kernel_weights()
        updates = 0
        for support in reversed(old_supports):
            if updates >= config.max_kernel_updates_per_event:
                break
            dt = event.ts - support.ts
            if dt < 0 or dt > config.active_window_sec:
                continue
            basis_values = [
                weight * _gaussian_rbf(dt, mean, bandwidth)
                for weight, mean, bandwidth in zip(
                    weights,
                    config.time_kernel_means_sec,
                    config.time_kernel_bandwidths_sec,
                )
            ]
            total = sum(basis_values)
            if total <= 0:
                continue
            for index, value in enumerate(basis_values):
                self.kernel_alpha[index] += value / total
            updates += 1

    def _update_topology_counts(self, event, old_supports, topology, config):
        updates = 0
        for support in reversed(old_supports):
            if updates >= config.max_kernel_updates_per_event:
                break
            relation = topology.relation(event, support, max_hops=config.topology_max_hops)
            self.topology_relation_counts[relation] += 1
            updates += 1


@dataclass
class _Particle:
    log_weight: float
    clusters: OrderedDict = field(default_factory=OrderedDict)
    next_cluster_number: int = 1
    last_proposal: _Proposal | None = None
    last_candidate_count: int = 0

    def new_cluster_id(self):
        cascade_id = f"cascade-{self.next_cluster_number}"
        self.next_cluster_number += 1
        return cascade_id


class TopologyPoweredDHP:
    """DHP-style particle filter for streaming alarm cascade membership.

    The local DHP reference selects a new versus active cluster with a
    Hawkes intensity and a Dirichlet-multinomial content likelihood. This
    implementation keeps that online structure, adds PDHP's powered time
    prior and multiplies it by a learned cluster-local topology affinity.
    """

    def __init__(self, config=None, topology=None):
        self.config = config or AlarmDHPConfig()
        self.topology = topology or TopologyIndex()
        self.rng = random.Random(self.config.seed)
        initial_log_weight = -math.log(self.config.particle_count)
        self.particles = [
            _Particle(log_weight=initial_log_weight)
            for _ in range(self.config.particle_count)
        ]
        self.vocabulary = set()
        self.last_ts = 0.0

    def observe_raise(self, event):
        self.last_ts = max(self.last_ts, event.ts)
        self.vocabulary.update(event.feature_counts)
        for particle in self.particles:
            self._step_particle(particle, event)
        self._normalize_weights()
        best = self._best_particle()
        proposal = best.last_proposal
        candidate_count = best.last_candidate_count
        decision = CascadeDecision(
            status="clustered",
            cascade_id=proposal.cluster_id,
            event=event,
            probability=proposal.probability,
            candidate_count=candidate_count,
            log_score=proposal.log_score,
            details={
                "time_rate": proposal.time_rate,
                "topology_affinity": proposal.topology_affinity,
                "content_log_score": proposal.content_log_score,
                "time_power": self.config.time_power,
            },
        )
        self._resample_if_needed()
        return decision

    def observe_clear(self, event):
        cleared_by_particle = []
        for particle in self.particles:
            cleared = [
                cascade_id
                for cascade_id, cluster in particle.clusters.items()
                if cluster.clear_event_key(event.event_key)
            ]
            cleared_by_particle.append(cleared)
        best_index = self.particles.index(self._best_particle())
        return CascadeDecision(
            status="clear",
            event=event,
            reason="clear_removed_from_active_alarm_keys",
            details={"cleared_cascades": cleared_by_particle[best_index]},
        )

    def cascade_snapshots(self, now_ts=None):
        now_ts = self.last_ts if now_ts is None else now_ts
        best = self._best_particle()
        return [
            best.clusters[cascade_id].snapshot(now_ts, self.config)
            for cascade_id in sorted(best.clusters, key=_cascade_sort_key)
        ]

    def cascade_count(self):
        return len(self._best_particle().clusters)

    def _step_particle(self, particle, event):
        proposals = [self._new_cluster_proposal(event)]
        for cluster in self._candidate_clusters(particle, event.ts):
            proposal = self._existing_cluster_proposal(cluster, event)
            if proposal is not None:
                proposals.append(proposal)

        log_normalizer = _logsumexp([proposal.log_score for proposal in proposals])
        probabilities = [
            math.exp(proposal.log_score - log_normalizer)
            for proposal in proposals
        ]
        for proposal, probability in zip(proposals, probabilities):
            proposal.probability = probability

        chosen_index = self._choose_proposal(probabilities)
        chosen = proposals[chosen_index]
        if chosen.cluster_id == "new":
            chosen.cluster_id = particle.new_cluster_id()
            cluster = _Cluster(
                cascade_id=chosen.cluster_id,
                created_ts=event.ts,
                last_ts=event.ts,
                kernel_alpha=[self.config.time_kernel_prior] * len(self.config.time_kernel_means_sec),
            )
            particle.clusters[chosen.cluster_id] = cluster
        else:
            cluster = particle.clusters[chosen.cluster_id]
        cluster.add(event, self.topology, self.config)
        particle.clusters.move_to_end(chosen.cluster_id)
        particle.log_weight += log_normalizer
        particle.last_proposal = chosen
        particle.last_candidate_count = len(proposals)

    def _candidate_clusters(self, particle, event_ts):
        """Yield the most recently updated active candidates first."""
        max_candidates = self.config.max_candidate_cascades
        yielded = 0
        for cascade_id in reversed(particle.clusters):
            cluster = particle.clusters[cascade_id]
            age = event_ts - cluster.last_ts
            if age < 0:
                continue
            if age > self.config.active_window_sec or age >= self.config.close_after_sec:
                # Bounded late events can perturb the recency order slightly.
                # Skip stale clusters without assuming every older slot is stale.
                continue
            if not cluster.is_candidate(event_ts, self.config):
                continue
            yield cluster
            yielded += 1
            if max_candidates and yielded >= max_candidates:
                break

    def _new_cluster_proposal(self, event):
        empty = _Cluster(
            cascade_id="new",
            created_ts=event.ts,
            last_ts=event.ts,
            kernel_alpha=[self.config.time_kernel_prior] * len(self.config.time_kernel_means_sec),
        )
        content_log_score = empty.content_log_predictive(
            event.feature_counts,
            len(self.vocabulary),
            self.config.theta0,
        )
        # PDHP leaves the base intensity as the new-cluster mass.
        log_score = math.log(self.config.base_intensity) + content_log_score
        return _Proposal(
            cluster_id="new",
            probability=0.0,
            log_score=log_score,
            time_rate=self.config.base_intensity,
            topology_affinity=1.0,
            content_log_score=content_log_score,
        )

    def _existing_cluster_proposal(self, cluster, event):
        time_rate = cluster.time_rate(event.ts, self.config)
        if time_rate <= 0:
            return None
        content_log_score = cluster.content_log_predictive(
            event.feature_counts,
            len(self.vocabulary),
            self.config.theta0,
        )
        topology_affinity = cluster.topology_affinity(event, self.topology, self.config)
        topology_affinity = max(topology_affinity, self.config.min_probability)
        time_rate = max(time_rate, self.config.min_probability)
        log_score = self.config.time_power * math.log(time_rate)
        log_score += content_log_score
        log_score += self.config.topology_strength * math.log(topology_affinity)
        return _Proposal(
            cluster_id=cluster.cascade_id,
            probability=0.0,
            log_score=log_score,
            time_rate=time_rate,
            topology_affinity=topology_affinity,
            content_log_score=content_log_score,
        )

    def _choose_proposal(self, probabilities):
        if self.config.assignment_strategy == "map" or len(probabilities) == 1:
            return max(range(len(probabilities)), key=lambda index: probabilities[index])
        draw = self.rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                return index
        return len(probabilities) - 1

    def _normalize_weights(self):
        normalizer = _logsumexp([particle.log_weight for particle in self.particles])
        for particle in self.particles:
            particle.log_weight -= normalizer

    def _best_particle(self):
        return max(self.particles, key=lambda particle: particle.log_weight)

    def _resample_if_needed(self):
        weights = [math.exp(particle.log_weight) for particle in self.particles]
        ess_denom = sum(weight * weight for weight in weights)
        effective_sample_size = 1.0 / ess_denom if ess_denom > 0 else 0.0
        if effective_sample_size >= self.config.resample_ess_ratio * len(self.particles):
            return

        cumulative = []
        running = 0.0
        for weight in weights:
            running += weight
            cumulative.append(running)
        new_particles = []
        for _ in self.particles:
            draw = self.rng.random()
            for index, upper_bound in enumerate(cumulative):
                if draw <= upper_bound:
                    new_particles.append(copy.deepcopy(self.particles[index]))
                    break
        uniform_log_weight = -math.log(len(new_particles))
        for particle in new_particles:
            particle.log_weight = uniform_log_weight
        self.particles = new_particles


def _gaussian_rbf(dt, mean, bandwidth):
    scaled = (dt - mean) / bandwidth
    return math.exp(-0.5 * scaled * scaled) / (math.sqrt(2.0 * math.pi) * bandwidth)


def _mixture_rbf(dt, weights, means, bandwidths):
    return sum(
        weight * _gaussian_rbf(dt, mean, bandwidth)
        for weight, mean, bandwidth in zip(weights, means, bandwidths)
    )


def _logsumexp(values):
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _cascade_sort_key(cascade_id):
    try:
        return int(str(cascade_id).rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return str(cascade_id)
