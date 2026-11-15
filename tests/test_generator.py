"""Tests for the config and the base signal generator."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aiops.simulator import BaseSignalConfig, generate_base_signal

SEED = 42


@pytest.fixture
def config():
    return BaseSignalConfig()


@pytest.fixture
def signal(config):
    return generate_base_signal(config, np.random.default_rng(SEED))


# Verifies that using the same random seed produces the same observed series.
def test_same_seed_gives_identical_series(config, signal):
    again = generate_base_signal(config, np.random.default_rng(SEED))
    np.testing.assert_array_equal(signal.observed, again.observed)


# Verifies that changing the random seed changes the generated noise.
def test_different_seed_gives_different_noise(config, signal):
    other = generate_base_signal(config, np.random.default_rng(SEED + 1))
    assert not np.array_equal(signal.noise, other.noise)


# Verifies the number of samples and five-minute timestamp spacing.
def test_length_and_spacing(signal):
    assert len(signal.timestamps) == 4032	# 14 days*288 samples per day
    for part in (signal.level, signal.seasonal, signal.noise):
        assert len(part) == 4032
    assert np.all(np.diff(signal.timestamps) == np.timedelta64(5, "m"))


# Verifies that default settings do not clip the expected signal plus noise.
def test_defaults_never_clip(signal):
    unclipped = signal.expected + signal.noise
    np.testing.assert_array_equal(signal.observed, unclipped)


# Verifies that the expected signal follows the configured sinusoidal formula.
def test_expected_matches_spec_formula(config, signal):
    hours = np.arange(config.n_steps) * config.interval_minutes / 60	# hours since the first midnight
    angle = 2 * np.pi * (hours - config.peak_hour) / 24
    spec = config.baseline + config.amplitude * np.sin(angle + np.pi / 2)	# the sine shifted to peak at peak_hour
    np.testing.assert_allclose(signal.expected, spec, atol=1e-9)


# Verifies that each day's expected signal peaks at the configured peak hour.
def test_every_day_peaks_at_peak_hour(signal):
    days = signal.expected.reshape(14, 288)	# one row per day
    peak_times = signal.timestamps.reshape(14, 288)[np.arange(14), days.argmax(axis=1)]
    minute_of_day = (peak_times - peak_times.astype("datetime64[D]")).astype(np.int64)
    assert np.all(minute_of_day == 14 * 60)


# Verifies the noise spread, near-zero mean, and configured lag-one correlation.
def test_noise_has_configured_spread_and_correlation(config, signal):
    noise = signal.noise
    lag1 = np.corrcoef(noise[:-1], noise[1:])[0, 1]	# correlation between each point and the next
    assert abs(noise.mean()) < 0.5
    assert noise.std() == pytest.approx(config.noise_std, rel=0.1)
    assert lag1 == pytest.approx(config.noise_ar_coef, abs=0.05)


# Verifies that an AR coefficient of zero produces uncorrelated noise.
def test_zero_ar_coef_gives_independent_noise():
    config = BaseSignalConfig(noise_ar_coef=0.0)
    noise = generate_base_signal(config, np.random.default_rng(SEED)).noise
    lag1 = np.corrcoef(noise[:-1], noise[1:])[0, 1]
    assert lag1 == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize(
    "bad_field",
    [
        {"noise_std": -1.0},
        {"period_minutes": 1441},	# not a multiple of the 5-minute interval
        {"noise_ar_coef": 1.0},	# AR(1) is only stable for |φ| < 1
        {"peak_hour": 24.0},
        {"baseline": 90.0},	# 90 + 15 + 3×3 = 114, past the 100% bound
    ],
)
# Verifies that invalid configuration values are rejected.
def test_invalid_config_is_rejected(bad_field):
    with pytest.raises(ValueError):
        BaseSignalConfig(**bad_field)


# Verifies that configuration instances cannot be modified after creation.
def test_config_is_frozen(config):
    with pytest.raises(FrozenInstanceError):
        config.baseline = 50.0