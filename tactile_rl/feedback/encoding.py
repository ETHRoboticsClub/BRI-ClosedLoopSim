"""ENCODING: convert normalized pressure to a desired vibration amplitude.

Contact hysteresis uses separate ON/OFF thresholds so tiny fluctuations
around contact don't repeatedly switch the motor on and off.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EncodingConfig:
    #the on off values can be adjusted as we test
    contact_on: float = 0.12
    contact_off: float = 0.06 #allows for hysteresis 
    amplitude_exponent: float = .5  # 1=linear, .5 is square root, etc

    #ensuring valid defintitions
    def __post_init__(self) -> None:
        if not (0 <= self.contact_off < self.contact_on <= 1):
            raise ValueError("Require 0 <= contact_off < contact_on <= 1")
        if self.amplitude_exponent <= 0:
            raise ValueError("amplitude_exponent must be positive")


class AmplitudeEncoder:
    def __init__(self, config: EncodingConfig | None = None):
        self.config = config or EncodingConfig()
        self.in_contact = False

    def update(self, normalized_force: float) -> float:
        """Return a [0, 1] commanded drive intensity (NOT measured vibration)."""
        if not math.isfinite(normalized_force):
            raise ValueError("normalized_force must be finite")
        x = min(1.0, max(0.0, normalized_force))

        if self.in_contact:
            if x <= self.config.contact_off:
                self.in_contact = False
        elif x >= self.config.contact_on:
            self.in_contact = True

        if not self.in_contact:
            return 0.0
        return x ** self.config.amplitude_exponent #where we actually translate the normalized force value

    def reset(self) -> None:
        self.in_contact = False
