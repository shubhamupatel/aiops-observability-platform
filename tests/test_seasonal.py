"""Tests for the seasonal z-score detector: its expected value, scale, and z-score on inputs with known answers, and that it only looks back."""

from pathlib import Path

import numpy as np
import pytest

from aiops.data import MetricSeries, load_synthetic_dataset
from aiops.detection import SeasonalZScoreDetector
from aiops.detection.seasonal import MAD_TO_SIGMA
from aiops.simulator import MetricConfig, SpikeConfig, generate_base_signal, inject_spike

SEED = 42
COMMITTED_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"	# the dataset in the repo
DEFAULT = SeasonalZScoreDetector()
SMALL = SeasonalZScoreDetector(period_minutes=20, history_periods=3, scale_window_periods=2, min_history_minutes=10)	# 4-sample days, 2-sample minimum
STAIRS = [10.0 * day + slot for day in range(5) for slot in range(4)]	# 5 days of 4 samples, each day 10 higher than the one before


def _series(values) -> MetricSeries:
    """A series at 5-minute intervals from 2026-01-05, the same timing as the simulator's."""
    values = np.array(values, dtype=np.float64)
    timestamps = np.datetime64("2026-01-05T00:00") + np.arange(len(values)) * np.timedelta64(5, "m")
    return MetricSeries("test", timestamps, values)


def _base_values() -> np.ndarray:
    """One normal simulated series: daily cycle and AR(1) noise, no anomalies."""
    return generate_base_signal(MetricConfig().base, np.random.default_rng(SEED)).observed


# Verifies the expected value: the median of the same time of day on up to 3 earlier days, as many as exist.
def test_expected_is_the_median_of_earlier_days():
    expected = SMALL.breakdown(_series(STAIRS)).expected
    assert np.isnan(expected[:4]).all()	# day 0 has no earlier day
    np.testing.assert_array_equal(
        expected[4:],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 20, 21, 22, 23],	# day 1 copies day 0, day 2 splits days 0 and 1, days 3 and 4 take the middle of 3
    )


# Verifies the scale: 1.4826 times the median absolute residual over the 8 samples before, once at least 2 exist.
def test_scale_is_the_median_absolute_residual_before_each_sample():
    scale = SMALL.breakdown(_series(STAIRS)).scale	# residuals are 10 on day 1, 15 on day 2, 20 on days 3 and 4
    assert np.isnan(scale[:6]).all()	# no residuals in day 0, then only one before sample 5
    np.testing.assert_allclose(scale[[6, 8, 12, 13]], MAD_TO_SIGMA * np.array([10, 10, 12.5, 15]))


# Verifies that z is the residual over the scale, and that the score is its size.
def test_z_is_residual_over_scale_and_score_is_its_size():
    breakdown = SMALL.breakdown(_series(STAIRS))
    assert breakdown.z[12] == pytest.approx(20 / (MAD_TO_SIGMA * 12.5))
    np.testing.assert_array_equal(SMALL.score(_series(STAIRS)), np.abs(breakdown.z))


# Verifies that one anomalous earlier day pulls the expected value halfway with 2 references, but not at all once 3 exist.
def test_one_bad_reference_day():
    values = np.zeros(20)
    values[4] = 100.0	# day 1, first slot
    expected = SMALL.breakdown(_series(values)).expected
    assert expected[8] == 50.0	# day 2: the median of 100 and 0
    assert expected[12] == 0.0	# day 3: the median of 100, 0, and 0


# Verifies a flat-lined history: a change from it scores infinite, and no change scores NaN, without any warning.
def test_flat_history():
    values = np.full(16, 10.0)
    values[13] = 11.0
    z = SMALL.breakdown(_series(values)).z
    assert z[13] == np.inf
    assert np.isnan(np.delete(z, 13)).all()


# Verifies that a series shorter than the look-back uses the earlier days it has.
def test_short_series_uses_the_days_it_has():
    expected = SMALL.breakdown(_series(STAIRS[:10])).expected	# 2.5 days, against a look-back of 3
    np.testing.assert_array_equal(expected[4:], [0, 1, 2, 3, 5, 6])


# Verifies that the detector only looks back: changing later values leaves every earlier result unchanged.
def test_changing_the_future_leaves_the_past_alone():
    series = load_synthetic_dataset(COMMITTED_DIR)[0].series
    changed = series.values.copy()
    changed[3000:] += 50.0
    before = DEFAULT.breakdown(series)
    after = DEFAULT.breakdown(MetricSeries(series.name, series.timestamps, changed))
    for name in ("expected", "residual", "scale", "z"):
        np.testing.assert_array_equal(getattr(after, name)[:3000], getattr(before, name)[:3000])
    assert not np.array_equal(after.residual[3000:], before.residual[3000:])	# the change did reach the later results


# Verifies the cold start on the real dataset: no score for the first day plus 6 hours, then a score at every sample.
def test_no_score_until_a_day_and_6_hours_of_history():
    z = DEFAULT.breakdown(load_synthetic_dataset(COMMITTED_DIR)[0].series).z
    assert np.flatnonzero(np.isnan(z)).tolist() == list(range(360))	# 288 samples in the first day, then 72 more


# Verifies that a +30 spike on a normal simulated series gets the series' highest score.
def test_spike_gets_the_highest_score():
    base = generate_base_signal(MetricConfig().base, np.random.default_rng(SEED))
    spiked, _ = inject_spike(base, 2000, SpikeConfig(magnitude_range=(30.0, 30.0)), np.random.default_rng(SEED))
    assert np.nanargmax(DEFAULT.score(_series(spiked.observed))) == 2000


# Verifies that z is positive while values sit above their usual level and negative while they sit below, and that both score high.
@pytest.mark.parametrize("shift", [15.0, -15.0])
def test_both_directions_score_high(shift):
    values = _base_values().copy()
    values[2000:2048] += shift	# 4 hours
    series = _series(values)
    assert np.sign(np.median(DEFAULT.breakdown(series).z[2000:2048])) == np.sign(shift)
    assert np.median(DEFAULT.score(series)[2000:2048]) > 3


# Verifies that settings that are zero, negative, or leave no room for a score are rejected.
@pytest.mark.parametrize(
    "settings",
    [
        dict(period_minutes=0),
        dict(history_periods=0),
        dict(scale_window_periods=-1),
        dict(min_history_minutes=0),
        dict(min_history_minutes=10081),	# a minute longer than the 7-day scale window
    ],
)
def test_detector_rejects_bad_settings(settings):
    with pytest.raises(ValueError):
        SeasonalZScoreDetector(**settings)


# Verifies that a period that isn't a whole number of samples is rejected instead of rounded.
def test_period_must_be_a_whole_number_of_samples():
    detector = SeasonalZScoreDetector(period_minutes=22, min_history_minutes=10)	# 22 minutes is 4.4 five-minute samples
    with pytest.raises(ValueError):
        detector.breakdown(_series(np.zeros(20)))


# Verifies that the breakdown's arrays can't be edited in place.
def test_breakdown_is_read_only():
    breakdown = SMALL.breakdown(_series(STAIRS))
    for array in (breakdown.expected, breakdown.residual, breakdown.scale, breakdown.z):
        with pytest.raises(ValueError):
            array[0] = 1.0