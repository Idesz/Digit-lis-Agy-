"""
Global State Vector & Homeostatic Controller.
Matches Pillar 3 from software_and_code_specification.md and technical_specification.md.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GlobalStateVector:
    """
    Homeostatic state vector.
    Attributes:
        curiosity: Curiosity drive level (0.0 - 1.0).
        error_rate: Error / surprise rate (0.0 - 1.0).
        energy_level: Energy and resource capacity (0.0 - 1.0).
        uncertainty: Epistemic doubt metric (0.0 - 1.0).
    """
    curiosity: float = 0.8
    error_rate: float = 0.05
    energy_level: float = 1.0
    uncertainty: float = 0.1


class HomeostaticController:
    """
    Evaluates internal physiological states and modulates cognitive behavior.
    """

    def __init__(self, state_vector: GlobalStateVector) -> None:
        self.state = state_vector

    def update_state(
        self,
        curiosity_delta: float = 0.0,
        error_delta: float = 0.0,
        energy_delta: float = 0.0,
        uncertainty_delta: float = 0.0,
    ) -> None:
        self.state.curiosity = max(0.0, min(1.0, self.state.curiosity + curiosity_delta))
        self.state.error_rate = max(0.0, min(1.0, self.state.error_rate + error_delta))
        self.state.energy_level = max(0.0, min(1.0, self.state.energy_level + energy_delta))
        self.state.uncertainty = max(0.0, min(1.0, self.state.uncertainty + uncertainty_delta))

    def evaluate_and_modulate(self) -> Dict[str, Any]:
        modulation_signals = {
            "trigger_safety_protocol": False,
            "generate_exploration_task": False,
            "reduce_compute_throughput": False,
            "recommended_system1_threshold": 0.80,
        }

        # High error rate or high uncertainty activates safety protocol
        if self.state.error_rate > 0.25 or self.state.uncertainty > 0.50:
            modulation_signals["trigger_safety_protocol"] = True
            modulation_signals["recommended_system1_threshold"] = 0.95

        # Low curiosity triggers exploration
        if self.state.curiosity < 0.20:
            modulation_signals["generate_exploration_task"] = True

        # Low energy reduces throughput
        if self.state.energy_level < 0.15:
            modulation_signals["reduce_compute_throughput"] = True

        return modulation_signals
