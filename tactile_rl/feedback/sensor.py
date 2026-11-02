"""SENSOR: turn MuJoCo contact force (newtons) into normalized [0, 1].

MuJoCo provides simulated physical force, NOT an FSR resistance or Arduino
ADC count
"""
from dataclasses import dataclass
import math


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class SensorConfig:
    min_force_n: float = 0.0       # calibrated force corresponding to zero
    max_force_n: float = 3.0       # initial estimate, what we will tune to be the normalized value of 1 (max)
    smoothing_alpha: float = 0.25  # EMA weight for NEW sample, 1 = no filtering

    def __post_init__(self) -> None:
        if not (0 <= self.min_force_n < self.max_force_n):
            raise ValueError("Force calibration requires 0 <= min < max")
        if not (0 < self.smoothing_alpha <= 1):
            raise ValueError("smoothing_alpha must be in (0, 1]")


@dataclass(frozen=True)
class SensorReading:
    force_n: float
    filtered_force_n: float
    normalized: float


class ForceSensor:
    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()
        self._filtered_n = 0.0

    def update(self, force_n: float) -> SensorReading:
        """Process one nonnegative contact force measurement in N."""
        #ensuring valid measurment from mujoco
        if not math.isfinite(force_n):
            raise ValueError("force_n must be finite")
        force_n = max(0.0, force_n)  # ignore any tiny numerical negative value
        a = self.config.smoothing_alpha
        #smooths based on the alpha value, alpha of new F value + 1-alpha of previous EFA
        #the stronger the smoothing (smaller the a) the less rapid changes, but more delay
        self._filtered_n = a * force_n + (1 - a) * self._filtered_n
        
        normalized = clamp01(
            (self._filtered_n - self.config.min_force_n)
            / (self.config.max_force_n - self.config.min_force_n)
        )
        return SensorReading(force_n, self._filtered_n, normalized)

    def reset(self) -> None:
        self._filtered_n = 0.0
