import math
from dataclasses import dataclass, replace

ARRIVAL_DISTRIBUTIONS = ("uniform", "exponential", "constant")
SERVICE_DISTRIBUTIONS = ("constant", "exponential")
CONTROL_MODES = ("adaptive", "never", "always")


@dataclass
class ModelConfig:
    arrival_min: float = 3.0
    arrival_max: float = 9.0
    arrival_mean: float = 6.0
    arrival_distribution: str = "uniform"
    service_time_normal: float = 5.0
    service_time_accelerated: float = 4.0
    service_distribution: str = "constant"
    destruction_threshold: float = 10.0
    time_tolerance: float = 1e-9
    destruction_limit: float = 0.30
    destruction_limit_window_size: int = 100
    window_size: int = 100
    activation_threshold: float = 0.30
    deactivation_threshold: float | None = None
    activate_at_threshold: bool = True
    deactivate_at_threshold: bool = False
    control_mode: str = "adaptive"
    early_discard: bool = False
    warmup_time: float = 0.0
    simulation_time: float = 10000.0
    seed: int | None = 1
    record_protocol: bool = False
    protocol_limit: int = 100
    trace_interval: float | None = None

    def __post_init__(self):
        if self.arrival_distribution not in ARRIVAL_DISTRIBUTIONS:
            raise ValueError(f"arrival_distribution must be one of {ARRIVAL_DISTRIBUTIONS}")
        if self.service_distribution not in SERVICE_DISTRIBUTIONS:
            raise ValueError(f"service_distribution must be one of {SERVICE_DISTRIBUTIONS}")
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}")
        if not 0 < self.arrival_min <= self.arrival_max:
            raise ValueError("arrival interval bounds must satisfy 0 < arrival_min <= arrival_max")
        if self.arrival_mean <= 0 or self.service_time_normal <= 0 or self.service_time_accelerated <= 0:
            raise ValueError("mean times must be positive")
        if self.service_time_accelerated > self.service_time_normal:
            raise ValueError("service_time_accelerated must not exceed service_time_normal")
        if self.destruction_threshold <= 0:
            raise ValueError("destruction_threshold must be positive")
        if self.time_tolerance < 0:
            raise ValueError("time_tolerance must be non-negative")
        if self.arrival_distribution == "uniform" and not math.isclose(self.arrival_mean, self.arrival_center):
            raise ValueError("for uniform arrivals arrival_mean must equal (arrival_min + arrival_max) / 2")
        if self.window_size < 1 or self.destruction_limit_window_size < 1:
            raise ValueError("window sizes must be at least 1")
        if not 0.0 <= self.destruction_limit <= 1.0:
            raise ValueError("destruction_limit must be within [0; 1]")
        if self.activation_threshold < 0.0:
            raise ValueError("activation_threshold must be non-negative")
        if self.effective_deactivation_threshold < 0.0:
            raise ValueError("deactivation_threshold must be non-negative")
        if self.effective_deactivation_threshold > self.activation_threshold:
            raise ValueError("deactivation_threshold must not exceed activation_threshold")
        if self.trace_interval is not None and self.trace_interval <= 0:
            raise ValueError("trace_interval must be positive or None")
        if self.warmup_time < 0 or self.simulation_time <= 0:
            raise ValueError("warmup_time must be non-negative and simulation_time positive")

    @property
    def effective_deactivation_threshold(self):
        if self.deactivation_threshold is None:
            return self.activation_threshold
        return self.deactivation_threshold

    @property
    def arrival_delay_mean(self):
        if self.arrival_distribution == "uniform":
            return self.arrival_center
        return self.arrival_mean

    @property
    def arrival_center(self):
        return (self.arrival_min + self.arrival_max) / 2

    @property
    def arrival_spread(self):
        return (self.arrival_max - self.arrival_min) / 2

    @property
    def end_time(self):
        return self.warmup_time + self.simulation_time

    def with_arrival_center(self, center, spread=None):
        if spread is None:
            spread = self.arrival_spread
        return replace(self, arrival_min=center - spread, arrival_max=center + spread, arrival_mean=center)

    def with_overrides(self, **overrides):
        return replace(self, **overrides)
