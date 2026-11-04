"""
Predictive Processing Unit and Residual Error Routing.
Matches Pillar 4 from software_and_code_specification.md.
"""

from typing import Dict, Tuple


class PredictiveProcessingUnit:
    """
    Predictive coding module evaluating residual surprise and routing signals.
    """

    def __init__(self, default_threshold: float = 0.20) -> None:
        self.default_threshold = default_threshold

    def predict_next_state(self, current_state: Dict[str, float]) -> Dict[str, float]:
        predicted_state = {}
        for key, value in current_state.items():
            predicted_state[key] = value * 1.02
        return predicted_state

    def calculate_residual_error(
        self,
        actual_input: Dict[str, float],
        predicted_input: Dict[str, float],
    ) -> float:
        total_error = 0.0
        count = 0
        for key in actual_input:
            if key in predicted_input:
                diff = abs(actual_input[key] - predicted_input[key])
                total_error += diff
                count += 1
        return total_error / count if count > 0 else 0.0

    def route_signal(self, residual_error: float, threshold: float) -> Tuple[bool, str]:
        if residual_error > threshold:
            return True, f"ERROR_EXCEEDS_THRESHOLD ({residual_error:.4f} > {threshold}): ROUTED_TO_UPPER_LAYERS"
        else:
            return False, f"ERROR_WITHIN_THRESHOLD ({residual_error:.4f} <= {threshold}): BYPASSED_UPPER_LAYERS"
