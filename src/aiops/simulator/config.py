"""MetricConfig: frozen dataclasses holding every simulator parameter."""

from dataclasses import dataclass, field
from datetime import datetime

MINUTES_PER_DAY = 24 * 60
NOISE_MARGIN_SIGMAS = 3.0	# noise stays within 3 standard deviations about 99.7% of the time


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

def _check_range(name: str, value_range: tuple[float, float]) -> None:
    """Raise unless value_range is (low, high) with 0 < low <= high."""
    low, high = value_range
    if not 0 < low <= high:
        raise ValueError(f"{name} must be (low, high) with 0 < low <= high, got {value_range}")


@dataclass(frozen=True)
class SpikeConfig:
    """Parameters for spikes (a one-sample upward jump)"""

    magnitude_range: tuple[float, float] = (10.0, 40.0)	# CPU points added, drawn uniformly from this range

    def __post_init__(self):
        _check_range("magnitude_range", self.magnitude_range)


@dataclass(frozen=True)
class LevelShiftConfig:
    """Parameters for level-shifts (a sustained upward step that drops back abruptly)"""

    magnitude_range: tuple[float, float] = (5.0, 25.0)	# CPU points added, drawn uniformly from this range
    duration_range_minutes: tuple[int, int] = (60, 480)	# how long the shift lasts, drawn uniformly (1 to 8 hours)

    def __post_init__(self):
        _check_range("magnitude_range", self.magnitude_range)
        _check_range("duration_range_minutes", self.duration_range_minutes)


@dataclass(frozen=True)
class MetricConfig:
    """Top-level simulator config (the base signal plus one sub-config per anomaly type)"""

    base: BaseSignalConfig = field(default_factory=BaseSignalConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    level_shift: LevelShiftConfig = field(default_factory=LevelShiftConfig)

    def __post_init__(self):
        self._check_headroom()
        self._check_level_shift_durations()

    def _check_headroom(self):
        upper = self.base.bounds[1]
        normal_peak = self.base.baseline + self.base.amplitude + NOISE_MARGIN_SIGMAS * self.base.noise_std
        largest_jumps = {	# anomalies never overlap, so each type only has to fit on its own
            "spike": self.spike.magnitude_range[1],
            "level_shift": self.level_shift.magnitude_range[1],
        }
        for name, jump in largest_jumps.items():
            if normal_peak + jump > upper:
                raise ValueError(
                    f"the largest {name} on top of the daily peak (+ {NOISE_MARGIN_SIGMAS:g}·noise_std) must stay within bounds"
                )

    def _check_level_shift_durations(self):
        interval = self.base.interval_minutes
        shortest, longest = self.level_shift.duration_range_minutes
        if shortest % interval or longest % interval:
            raise ValueError(f"level_shift durations must be multiples of interval_minutes ({interval})")
        if longest > self.base.duration_days * MINUTES_PER_DAY:
            raise ValueError("level_shift's longest duration must fit inside the series")