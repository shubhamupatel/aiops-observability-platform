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


def _check_count(count: int) -> None:
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")


@dataclass(frozen=True)
class SpikeConfig:
    """Parameters for spikes: a one-sample upward jump."""

    magnitude_range: tuple[float, float] = (10.0, 40.0)	# CPU points added, drawn uniformly from this range
    count: int = 3	# spikes per series; 0 turns spikes off

    def __post_init__(self):
        _check_range("magnitude_range", self.magnitude_range)
        _check_count(self.count)


@dataclass(frozen=True)
class LevelShiftConfig:
    """Parameters for level-shifts: a sustained upward step that drops back abruptly."""

    magnitude_range: tuple[float, float] = (5.0, 25.0)	# CPU points added, drawn uniformly from this range
    duration_range_minutes: tuple[int, int] = (60, 480)	# how long the shift lasts, drawn uniformly (1 to 8 hours)
    count: int = 2	# level-shifts per series; 0 turns them off

    def __post_init__(self):
        _check_range("magnitude_range", self.magnitude_range)
        _check_range("duration_range_minutes", self.duration_range_minutes)
        _check_count(self.count)


@dataclass(frozen=True)
class SeasonalSuppressionConfig:
    """Parameters for seasonal suppression: the daily cycle disappears for a while, then returns."""

    duration_range_minutes: tuple[int, int] = (720, 1440)	# 12 to 24 hours, so every window includes a daily peak or low
    count: int = 1	# suppressions per series; 0 turns them off

    def __post_init__(self):
        _check_range("duration_range_minutes", self.duration_range_minutes)
        _check_count(self.count)


@dataclass(frozen=True)
class DriftConfig:
    """Parameters for drift: a slow linear climb that drops back abruptly, like a memory leak ended by a restart."""

    magnitude_range: tuple[float, float] = (10.0, 40.0)	# total CPU points the ramp climbs, drawn uniformly
    duration_range_minutes: tuple[int, int] = (360, 1440)	# how long the ramp lasts, drawn uniformly (6 to 24 hours)
    label_threshold: float = 5.0	# samples count as anomalous once the drift has added at least this many points
    count: int = 1	# drifts per series; 0 turns them off

    def __post_init__(self):
        _check_range("magnitude_range", self.magnitude_range)
        _check_range("duration_range_minutes", self.duration_range_minutes)
        if not 0 < self.label_threshold < self.magnitude_range[0]:
            raise ValueError(
                f"label_threshold must be positive and below the smallest magnitude, got {self.label_threshold}"
            )
        _check_count(self.count)


@dataclass(frozen=True)
class ScheduleConfig:
    """Rules for placing anomalies in a series: a quiet warm-up, gaps between anomalies, and a retry limit."""

    warmup_minutes: int = 1440	# no anomalies in the first day, so detectors see one full normal cycle first
    min_gap_minutes: int = 120	# normal data required between any two anomalies
    max_attempts: int = 100	# placement tries per anomaly before giving up with an error

    def __post_init__(self):
        if self.warmup_minutes < 0 or self.min_gap_minutes < 0:
            raise ValueError("warmup_minutes and min_gap_minutes must be non-negative")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")


@dataclass(frozen=True)
class MetricConfig:
    """Top-level simulator config: the base signal, one sub-config per anomaly type, and the placement rules."""

    base: BaseSignalConfig = field(default_factory=BaseSignalConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    level_shift: LevelShiftConfig = field(default_factory=LevelShiftConfig)
    seasonal_suppression: SeasonalSuppressionConfig = field(default_factory=SeasonalSuppressionConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    def __post_init__(self):
        self._check_headroom()
        self._check_durations()
        self._check_schedule()

    def _check_headroom(self):
        upper = self.base.bounds[1]
        normal_peak = self.base.baseline + self.base.amplitude + NOISE_MARGIN_SIGMAS * self.base.noise_std
        largest_jumps = {	# anomalies never overlap, so each type only has to fit on its own
            "spike": self.spike.magnitude_range[1],
            "level_shift": self.level_shift.magnitude_range[1],
            "drift": self.drift.magnitude_range[1],
        }	# seasonal_suppression isn't listed: flattening the cycle moves values toward the baseline, never past a bound
        for name, jump in largest_jumps.items():
            if normal_peak + jump > upper:
                raise ValueError(
                    f"the largest {name} on top of the daily peak (+ {NOISE_MARGIN_SIGMAS:g}·noise_std) must stay within bounds"
                )

    def _check_durations(self):
        interval = self.base.interval_minutes
        series_minutes = self.base.duration_days * MINUTES_PER_DAY
        duration_ranges = {
            "level_shift": self.level_shift.duration_range_minutes,
            "seasonal_suppression": self.seasonal_suppression.duration_range_minutes,
            "drift": self.drift.duration_range_minutes,
        }
        for name, (shortest, longest) in duration_ranges.items():
            if shortest % interval or longest % interval:
                raise ValueError(f"{name} durations must be multiples of interval_minutes ({interval})")
            if longest > series_minutes:
                raise ValueError(f"{name}'s longest duration must fit inside the series")

    def _check_schedule(self):
        interval = self.base.interval_minutes
        if self.schedule.warmup_minutes % interval or self.schedule.min_gap_minutes % interval:
            raise ValueError(f"warmup_minutes and min_gap_minutes must be multiples of interval_minutes ({interval})")
        longest_and_count = [	# (longest duration in minutes, count) for each type; a spike lasts one sample
            (interval, self.spike.count),
            (self.level_shift.duration_range_minutes[1], self.level_shift.count),
            (self.seasonal_suppression.duration_range_minutes[1], self.seasonal_suppression.count),
            (self.drift.duration_range_minutes[1], self.drift.count),
        ]
        worst_case = self.schedule.warmup_minutes + sum(
            count * (longest + self.schedule.min_gap_minutes) for longest, count in longest_and_count
        )
        if worst_case > self.base.duration_days * MINUTES_PER_DAY:
            raise ValueError(
                f"warm-up plus every anomaly at its longest, each with its gap, needs {worst_case} minutes; the series has fewer"
            )