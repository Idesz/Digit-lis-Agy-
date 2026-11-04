"""
JEPA Sandbox World Model and Virtual Rollout Simulator.
Matches Pillar 6 from software_and_code_specification.md.
"""

from typing import Any, Dict, List


class JEPASandboxWorldModel:
    """
    JEPA abstract latent simulation sandbox evaluating multi-step virtual rollouts.
    Enforces candidate action constraint (between 5 and 10 scenarios).
    """

    def __init__(self) -> None:
        self.current_abstract_state: Dict[str, Any] = {}

    def abstract_state_encoder(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        abstract_representation = {}
        for key, value in raw_state.items():
            abstract_representation[f"abstract_{key}"] = hash(str(value)) % 1000

        self.current_abstract_state = abstract_representation
        return abstract_representation

    def virtual_rollout(self, action_candidates: List[str], depth: int = 10) -> List[Dict[str, Any]]:
        num_candidates = len(action_candidates)
        if not (5 <= num_candidates <= 10):
            raise ValueError(
                f"A virtuális rollout szigorúan 5 és 10 közötti forgatókönyv-jelöltet igényel. "
                f"Kapott elemszám: {num_candidates}"
            )

        simulated_outcomes = []

        for action in action_candidates:
            simulated_state = self.current_abstract_state.copy()
            risk_score = 0.0
            utility_score = 0.0

            for step in range(1, depth + 1):
                simulated_state["step"] = step
                if "risk" in action.lower() or "unsafe" in action.lower():
                    risk_score += 0.20 * step
                else:
                    risk_score += 0.02 * step

                utility_score += 0.15 * step

            net_score = utility_score - risk_score
            simulated_outcomes.append({
                "action": action,
                "final_abstract_state": simulated_state,
                "risk_score": risk_score,
                "utility_score": utility_score,
                "net_score": net_score,
            })

        return simulated_outcomes

    def evaluate_and_select_path(self, simulated_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not simulated_outcomes:
            raise ValueError("Nincsenek szimulált kimenetelek a kiértékeléshez.")

        best_outcome = max(simulated_outcomes, key=lambda x: x["net_score"])
        return best_outcome
