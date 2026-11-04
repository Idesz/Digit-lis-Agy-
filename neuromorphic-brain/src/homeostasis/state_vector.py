"""
Global State Vector & Homeostatic Control Module.
Tracks the 4-dimensional dynamic state: [curiosity, error_rate, resource_stress, uncertainty].
Dynamically modulates agent behavior:
- High curiosity -> Data acquisition / exploration mode
- High error_rate or uncertainty -> Cautious / deliberative decision making
- High resource_stress -> Resource conservation mode
"""

from typing import List, Tuple
from enum import Enum
import numpy as np


class BehavioralMode(str, Enum):
    EXPLORATION = "EXPLORATION"
    CAUTIOUS = "CAUTIOUS"
    CONSERVATION = "CONSERVATION"
    BALANCED = "BALANCED"


class HomeostaticState:
    """
    Maintains and regulates the agent's internal physiological-like balance.
    State vector: [curiosity, error_rate, resource_stress, uncertainty] in [0.0, 1.0]^4.
    """

    def __init__(
        self,
        curiosity: float = 0.5,
        error_rate: float = 0.1,
        resource_stress: float = 0.1,
        uncertainty: float = 0.2,
        decay_rate: float = 0.05,
    ):
        self.curiosity = float(np.clip(curiosity, 0.0, 1.0))
        self.error_rate = float(np.clip(error_rate, 0.0, 1.0))
        self.resource_stress = float(np.clip(resource_stress, 0.0, 1.0))
        self.uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        self.decay_rate = decay_rate

    @property
    def vector(self) -> List[float]:
        return [self.curiosity, self.error_rate, self.resource_stress, self.uncertainty]

    def update_observation(
        self,
        surprise: float,
        task_error: float,
        system_load: float,
        entropy: float,
        smoothing: float = 0.2,
    ) -> "HomeostaticState":
        """
        Updates internal state using moving averages and novelty feedback:
        - surprise increases curiosity
        - task_error drives error_rate
        - system_load drives resource_stress
        - entropy drives uncertainty
        """
        target_curiosity = np.clip(surprise * 1.2, 0.0, 1.0)
        target_error = np.clip(task_error, 0.0, 1.0)
        target_stress = np.clip(system_load, 0.0, 1.0)
        target_uncertainty = np.clip(entropy, 0.0, 1.0)

        self.curiosity = float((1.0 - smoothing) * self.curiosity + smoothing * target_curiosity)
        self.error_rate = float((1.0 - smoothing) * self.error_rate + smoothing * target_error)
        self.resource_stress = float((1.0 - smoothing) * self.resource_stress + smoothing * target_stress)
        self.uncertainty = float((1.0 - smoothing) * self.uncertainty + smoothing * target_uncertainty)

        return self

    def step_decay(self) -> None:
        """Passive decay towards baseline resting state."""
        self.curiosity = float(max(0.1, self.curiosity - self.decay_rate * 0.5))
        self.error_rate = float(max(0.05, self.error_rate - self.decay_rate))
        self.resource_stress = float(max(0.05, self.resource_stress - self.decay_rate))
        self.uncertainty = float(max(0.05, self.uncertainty - self.decay_rate))

    def get_behavioral_mode(
        self,
        curiosity_thresh: float = 0.6,
        error_thresh: float = 0.4,
        stress_thresh: float = 0.7,
        uncertainty_thresh: float = 0.5,
    ) -> BehavioralMode:
        """
        Determines high-level behavioral mode based on homeostatic pressures:
        - CONSERVATION: High resource stress takes highest priority.
        - CAUTIOUS: High error rate or high uncertainty slows down responses.
        - EXPLORATION: High curiosity with manageable error induces data seeking.
        - BALANCED: Default nominal state.
        """
        if self.resource_stress > stress_thresh:
            return BehavioralMode.CONSERVATION
        if self.error_rate > error_thresh or self.uncertainty > uncertainty_thresh:
            return BehavioralMode.CAUTIOUS
        if self.curiosity > curiosity_thresh and self.error_rate < error_thresh:
            return BehavioralMode.EXPLORATION
        return BehavioralMode.BALANCED
