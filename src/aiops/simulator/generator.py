"""Base signal = baseline + daily cycle + AR(1) Gaussian noise, clipped to bounds [0-100]."""

from dataclasses import dataclass

import numpy as np

from aiops.simulator.config import BaseSignalConfig


@dataclass
class SignalComponents:
    """Container dataclass that holds the separate components of a generated time-series signal.

    Parts are kept separate so injectors and labels each use the part they need.
    """

    timestamps: np.ndarray	# array of time markers (datetime64[m]) spaced at regular intervals
    level: np.ndarray	# the baseline; level-shifts and drift will edit this
    seasonal: np.ndarray	# the daily cycle; seasonal suppression will edit this
    noise: np.ndarray	# AR(1) noise
    bounds: tuple[float, float]

    @property
    def expected(self) -> np.ndarray:
        """Noise-free value from the true formula. Labels use this."""
        return self.level + self.seasonal

    @property
    def observed(self) -> np.ndarray:
        """What a monitoring system would record. Detectors use this."""
        low, high = self.bounds
        return np.clip(self.expected + self.noise, low, high)


def generate_base_signal(config: BaseSignalConfig, rng: np.random.Generator) -> SignalComponents:
    """Build one anomaly-free series from the config, drawing all randomness from rng."""
    timestamps = _make_timestamps(config)
    return SignalComponents(
        timestamps=timestamps,
        level=np.full(config.n_steps, float(config.baseline)),
        seasonal=_make_seasonal(config, timestamps),
        noise=_make_ar1_noise(config, rng),
        bounds=config.bounds,
    )


def _make_timestamps(config: BaseSignalConfig) -> np.ndarray:
    start = np.datetime64(config.start, "m")	# minute precision
    offsets = np.arange(config.n_steps) * np.timedelta64(config.interval_minutes, "m")
    return start + offsets


def _make_seasonal(config: BaseSignalConfig, timestamps: np.ndarray) -> np.ndarray:
    midnight = timestamps[0].astype("datetime64[D]")	# midnight of the first day
    minutes = (timestamps - midnight).astype(np.int64)	# minutes since that midnight
    angle = 2 * np.pi * (minutes - config.peak_hour * 60) / config.period_minutes
    return config.amplitude * np.cos(angle)	# cos peaks where angle = 0, i.e. at peak_hour


def _make_ar1_noise(config: BaseSignalConfig, rng: np.random.Generator) -> np.ndarray:
    phi = config.noise_ar_coef
    eps = rng.standard_normal(config.n_steps)	# all randomness drawn in one call
    innovation_std = config.noise_std * np.sqrt(1.0 - phi**2)	# keeps the overall std at noise_std
    noise = np.empty(config.n_steps)
    noise[0] = config.noise_std * eps[0]	# start already at the long-run spread
    for t in range(1, config.n_steps):
        noise[t] = phi * noise[t - 1] + innovation_std * eps[t]
    return noise