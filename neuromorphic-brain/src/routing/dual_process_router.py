"""
Dual-Process Architecture (System 1 vs System 2 Gating).
Matches Pillar 5 from software_and_code_specification.md.
"""

from typing import Any, Dict, Optional


class System1FastEngine:
    """Lightweight, low-latency pattern matching reflex (System 1)."""

    def generate_fast_response(self, task: str) -> Dict[str, Any]:
        is_complex = "complex" in task.lower() or "hard" in task.lower()
        confidence = 0.60 if is_complex else 0.92
        return {
            "response": f"System 1 intuitive output for: {task}",
            "confidence": confidence,
        }


class ConfidenceEvaluator:
    """Evaluates System 1 confidence against default or homeostatically modulated thresholds."""

    def __init__(self, default_threshold: float = 0.80) -> None:
        self.default_threshold = default_threshold

    def is_confident(self, confidence_score: float, custom_threshold: Optional[float] = None) -> bool:
        threshold = custom_threshold if custom_threshold is not None else self.default_threshold
        return confidence_score >= threshold


class System2SlowPlanner:
    """Deliberate MCTS and symbolic planner (System 2)."""

    def plan_and_solve(self, task: str) -> Dict[str, Any]:
        return {
            "response": f"System 2 deliberate MCTS solution for: {task}",
            "mcts_depth_reached": 10,
            "confidence": 0.99,
        }


class DualProcessRouter:
    """
    Gating mechanism routing between System 1 and System 2 with homeostatic threshold overrides.
    """

    def __init__(self, default_threshold: float = 0.80) -> None:
        self.system1 = System1FastEngine()
        self.evaluator = ConfidenceEvaluator(default_threshold=default_threshold)
        self.system2 = System2SlowPlanner()

    def process_input(self, task: str, dynamic_threshold_override: Optional[float] = None) -> Dict[str, Any]:
        fast_result = self.system1.generate_fast_response(task)

        if self.evaluator.is_confident(fast_result["confidence"], custom_threshold=dynamic_threshold_override):
            return {
                "execution_path": "SYSTEM_1_FAST",
                "output": fast_result["response"],
                "confidence": fast_result["confidence"],
            }

        slow_result = self.system2.plan_and_solve(task)
        return {
            "execution_path": "SYSTEM_2_SLOW",
            "output": slow_result["response"],
            "confidence": slow_result["confidence"],
        }
