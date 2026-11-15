"""MetricConfig: frozen dataclasses holding every simulator parameter."""

from dataclasses import dataclass, field
from datetime import datetime

MINUTES_PER_DAY = 24 * 60
NOISE_MARGIN_SIGMAS = 3.0	# noise stays within ±3σ about 99.7% of the time


@dataclass(frozen=True)
class BaseSignalConfig:
    """Parameters for the normal, anomaly-free CPU signal."""

    baseline: float = 35.0	# average CPU %, before the daily cycle
    amplitude: float = 15.0	# how far the daily peak rises above the baseline
    period_minutes: int = 1440	# length of one seasonal cycle (one day)
    peak_hour: float = 14.0	# hour of day the cycle peaks (2 p.m.)
    interval_minutes: int = 5	# time between samples
    duration_days: int = 14	# length of the series
    noise_std: float = 3.0	# overall standard deviation of the noise
    noise_ar_coef: float = 0.6	# AR(1) coefficient φ; 0 means independent noise
    bounds: tuple[float, float] = (0.0, 100.0)	# physical limits of CPU %
    start: datetime = datetime(2026, 1, 5)	# first timestamp, at midnight, treated as UTC

    def __post_init__(self):
        low, high = self.bounds
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        if self.period_minutes <= 0 or self.period_minutes % self.interval_minutes != 0:
            raise ValueError("period_minutes must be a positive multiple of interval_minutes")
        if self.duration_days <= 0 or (self.duration_days * MINUTES_PER_DAY) % self.interval_minutes != 0:
            raise ValueError("duration_days must be positive and a whole number of intervals long")
        if not 0 <= self.peak_hour < 24:
            raise ValueError("peak_hour must be in [0, 24)")
        if self.amplitude < 0 or self.noise_std < 0:
            raise ValueError("amplitude and noise_std must be non-negative")
        if not -1 < self.noise_ar_coef < 1:
            raise ValueError("noise_ar_coef must be in (-1, 1), or the noise grows without limit")
        if low >= high:
            raise ValueError("bounds must be (low, high) with low < high")
        swing = self.amplitude + NOISE_MARGIN_SIGMAS * self.noise_std
        if self.baseline - swing < low or self.baseline + swing > high:
            raise ValueError(f"baseline ± (amplitude + {NOISE_MARGIN_SIGMAS:g}·noise_std) must stay within bounds")

    @property
    def n_steps(self) -> int:
        """Number of samples in the series."""
        return self.duration_days * MINUTES_PER_DAY // self.interval_minutes


@dataclass(frozen=True)
class MetricConfig:
    """Top-level simulator config; the anomaly sub-configs join it in the injector steps."""

    base: BaseSignalConfig = field(default_factory=BaseSignalConfig)