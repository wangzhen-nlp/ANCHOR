from dataclasses import dataclass, field


def _relation_weights():
    # Weights are priors for online cluster-local topology affinities.
    return {
        "same_device": 1.0,
        "ne_hop_1": 0.92,
        "ne_hop_2": 0.72,
        "ne_hop_far": 0.48,
        # Site/domain fallbacks stay soft when the NE graph has no support.
        "same_site": 0.64,
        "hop_1": 0.72,
        "hop_2": 0.48,
        "hop_far": 0.28,
        "same_domain": 0.32,
        "disconnected": 0.12,
    }


@dataclass
class AlarmDHPConfig:
    """Configuration for alarm cascade clustering.

    Time defaults are seconds and intentionally focus on short alarm bursts
    plus a longer tail. They should be calibrated with local incident history.
    """

    particle_count: int = 4
    seed: int = 1024
    assignment_strategy: str = "map"
    theta0: float = 0.08
    base_intensity: float = 0.0001
    time_power: float = 1.0
    time_kernel_means_sec: tuple = (1.0, 5.0, 30.0, 120.0, 600.0, 1800.0)
    time_kernel_bandwidths_sec: tuple = (1.5, 4.0, 20.0, 90.0, 360.0, 900.0)
    time_kernel_prior: float = 0.6
    active_window_sec: float = 7200.0
    cooling_after_sec: float = 1800.0
    close_after_sec: float = 7200.0
    max_candidate_cascades: int = 1024
    max_support_events: int = 256
    max_kernel_updates_per_event: int = 32
    # EMA decay applied to kernel_alpha before each event's accumulation. Keeps
    # long-lived cascades' temporal kernels responsive to recent dynamics; the
    # default is small enough to be invisible on short streams.
    kernel_decay_per_event: float = 0.001
    topology_strength: float = 1.0
    topology_prior_mass: float = 3.0
    topology_max_hops: int = 2
    require_topology_candidate: bool = False
    topology_context_hops: int = 1
    topology_context_limit: int = 16
    topology_relation_weights: dict = field(default_factory=_relation_weights)
    resample_ess_ratio: float = 0.55
    min_probability: float = 1e-300

    def __post_init__(self):
        if self.particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if self.theta0 <= 0 or self.base_intensity <= 0:
            raise ValueError("theta0 and base_intensity must be positive")
        if self.time_power < 0:
            raise ValueError("time_power must be non-negative")
        if len(self.time_kernel_means_sec) != len(self.time_kernel_bandwidths_sec):
            raise ValueError("time kernel means and bandwidths must have the same length")
        if not self.time_kernel_means_sec:
            raise ValueError("at least one time kernel is required")
        if any(value <= 0 for value in self.time_kernel_bandwidths_sec):
            raise ValueError("time kernel bandwidths must be positive")
        if self.assignment_strategy not in {"sample", "map"}:
            raise ValueError("assignment_strategy must be 'sample' or 'map'")
        if self.active_window_sec <= 0 or self.close_after_sec <= 0:
            raise ValueError("active and close windows must be positive")
        if self.max_candidate_cascades < 0:
            raise ValueError("max_candidate_cascades must be non-negative")
        if self.topology_max_hops < 0:
            raise ValueError("topology_max_hops must be non-negative")
        if not 0.0 <= self.kernel_decay_per_event < 1.0:
            raise ValueError("kernel_decay_per_event must be in [0.0, 1.0)")


@dataclass
class StreamPolicyConfig:
    """Input-stream hygiene before alarms reach the DHP model."""

    reorder_lag_sec: float = 300.0
    late_tolerance_sec: float = 30.0
    duplicate_window_sec: float = 120.0
    flap_window_sec: float = 300.0
    emit_orphan_clears: bool = False
    debug_skips: bool = False

    def __post_init__(self):
        for name in (
            "reorder_lag_sec",
            "late_tolerance_sec",
            "duplicate_window_sec",
            "flap_window_sec",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
