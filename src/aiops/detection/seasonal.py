"""Seasonal z-score detector: compares each value with the same time of day on recent days, in units of a typical deviation."""

from dataclasses import dataclass

import numpy as np

from aiops.data import MetricSeries

MAD_TO_SIGMA = 1.4826	# for normal noise the median absolute deviation is 0.6745 standard deviations; this undoes that


@dataclass(frozen=True, eq=False)
class SeasonalBreakdown:
    """The detector's working at every sample, so a flag can be explained: what was expected, how far off the value was, and how unusual that is."""

    expected: np.ndarray	# median of the same time of day on recent days; NaN in the first day
    residual: np.ndarray	# observed minus expected, in the metric's own units
    scale: np.ndarray	# typical size of a recent residual, in the metric's units; NaN until there's enough history
    z: np.ndarray	# residual over scale: positive above the usual level, negative below, NaN where the detector can't judge

    def __post_init__(self):
        for array in (self.expected, self.residual, self.scale, self.z):
            array.flags.writeable = False	# read by explanations later, so no in-place edits


@dataclass(frozen=True)
class SeasonalZScoreDetector:
    """Flags values far from their usual level at that time of day. Its settings are fixed by reasoning; only the threshold on its scores gets tuned."""

    period_minutes: int = 1440	# one daily cycle
    history_periods: int = 7	# expected value from up to 7 earlier days; once 3 exist, one bad day can't move the median
    scale_window_periods: int = 7	# typical deviation from the last 7 days of residuals, so even a 24-hour anomaly is under 15% of them
    min_history_minutes: int = 360	# no score until 6 hours of residuals exist; fewer correlated samples give too noisy a scale

    def __post_init__(self):
        if min(self.period_minutes, self.history_periods, self.scale_window_periods, self.min_history_minutes) <= 0:
            raise ValueError("every setting must be positive")
        if self.min_history_minutes > self.scale_window_periods * self.period_minutes:
            raise ValueError("min_history_minutes can't be longer than the scale window, or no score could ever be computed")

    def score(self, series: MetricSeries) -> np.ndarray:
        """Score a series: how many typical deviations each value is from its usual level, either way. NaN where there's too little history to judge."""
        return np.abs(self.breakdown(series).z)

    def breakdown(self, series: MetricSeries) -> SeasonalBreakdown:
        """Work out the expected value, residual, typical deviation, and z-score at every sample, using only data from before that sample."""
        interval = series.interval_minutes
        period = _to_samples(self.period_minutes, interval)
        expected = _seasonal_expected(series.values, period, self.history_periods)
        residual = series.values - expected
        scale = _trailing_scale(residual, period * self.scale_window_periods, _to_samples(self.min_history_minutes, interval))
        with np.errstate(divide="ignore", invalid="ignore"):	# a flat-lined history has scale 0: any change from it scores infinite, no change scores NaN
            z = residual / scale
        return SeasonalBreakdown(expected=expected, residual=residual, scale=scale, z=z)


def _to_samples(minutes: int, interval: float) -> int:
    samples = minutes / interval
    if samples != int(samples):
        raise ValueError(f"{minutes} minutes isn't a whole number of {interval:g}-minute samples")
    return int(samples)


def _seasonal_expected(values: np.ndarray, period: int, history_periods: int) -> np.ndarray:
    """Median of the values exactly 1, 2, ... history_periods periods earlier, as many as exist. NaN where none do."""
    n_steps = len(values)
    earlier = np.full((history_periods, n_steps), np.nan)	# row k holds the value k + 1 periods before each sample
    for k in range(history_periods):
        lag = (k + 1) * period
        if lag >= n_steps:
            break	# the series is too short to look back this far, or any further
        earlier[k, lag:] = values[:n_steps - lag]
    expected = np.full(n_steps, np.nan)
    known = ~np.isnan(earlier).all(axis=0)	# samples with at least one earlier period to compare against
    expected[known] = np.nanmedian(earlier[:, known], axis=0)
    return expected


def _trailing_scale(residual: np.ndarray, window: int, minimum: int) -> np.ndarray:
    """Typical residual size over the window before each sample, scaled to match a standard deviation. NaN until minimum residuals exist."""
    size = np.abs(residual)	# about 0, not about their own median, since residuals are already deviations from the expected value
    scale = np.full(len(residual), np.nan)
    for t in range(len(residual)):
        past = size[max(0, t - window):t]	# strictly before t, so a value never sets its own yardstick
        past = past[~np.isnan(past)]	# the first period has no residuals
        if len(past) >= minimum:
            scale[t] = MAD_TO_SIGMA * np.median(past)
    return scale