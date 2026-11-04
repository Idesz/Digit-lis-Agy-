"""
Cognitive Decision Routing (CDR) Module.
Dynamic dual-process arbitration between:
- System 1: Fast, instinctive, local SNN feedforward reflex.
- System 2: Slow, deliberate, structured semantic inference and rollout simulation.

Evaluates 4 core dimensions:
1. C_s: Correlation strength with past memories.
2. D_c: Domain crossing / cross-domain complexity.
3. S_m: Stakeholder / entity involvement metric.
4. U_l: Uncertainty level.
Gating rule: R(q) = FAST if f(C_s, D_c, S_m, U_l) < tau else SLOW.
"""

from typing import Dict, Tuple
from enum import Enum
import numpy as np


class SystemRoute(str, Enum):
    FAST = "FAST_SYSTEM_1"
    SLOW = "SLOW_SYSTEM_2"


class CDRRouter:
    """
    Cognitive Decision Router determining execution paths based on task complexity and doubt.
    """

    def __init__(
        self,
        tau: float = 0.5,
        w_correlation: float = 0.3,
        w_domain: float = 0.25,
        w_stakeholders: float = 0.2,
        w_uncertainty: float = 0.25,
    ):
        self.tau = tau
        self.w_correlation = w_correlation
        self.w_domain = w_domain
        self.w_stakeholders = w_stakeholders
        self.w_uncertainty = w_uncertainty

        # Normalize weights
        total_w = w_correlation + w_domain + w_stakeholders + w_uncertainty
        self.w_correlation /= total_w
        self.w_domain /= total_w
        self.w_stakeholders /= total_w
        self.w_uncertainty /= total_w

    def compute_complexity_metric(
        self,
        correlation_strength: float,
        domain_crossing: float,
        stakeholder_count: int,
        uncertainty: float,
        max_stakeholders: int = 10,
    ) -> float:
        """
        f(C_s, D_c, S_m, U_l):
        - High correlation strength reduces complexity (familiar experience).
        - High domain crossing increases complexity.
        - Higher number of stakeholders increases complexity.
        - High uncertainty increases complexity.
        """
        c_s = np.clip(correlation_strength, 0.0, 1.0)
        d_c = np.clip(domain_crossing, 0.0, 1.0)
        s_m = np.clip(stakeholder_count / max(1, max_stakeholders), 0.0, 1.0)
        u_l = np.clip(uncertainty, 0.0, 1.0)

        unfamiliarity = 1.0 - c_s

        score = (
            self.w_correlation * unfamiliarity
            + self.w_domain * d_c
            + self.w_stakeholders * s_m
            + self.w_uncertainty * u_l
        )
        return float(score)

    def route(
        self,
        correlation_strength: float,
        domain_crossing: float,
        stakeholder_count: int,
        uncertainty: float,
    ) -> Tuple[SystemRoute, float, Dict[str, float]]:
        """
        Routes the task or perception query to FAST (System 1) or SLOW (System 2).
        """
        complexity = self.compute_complexity_metric(
            correlation_strength=correlation_strength,
            domain_crossing=domain_crossing,
            stakeholder_count=stakeholder_count,
            uncertainty=uncertainty,
        )

        decision = SystemRoute.FAST if complexity < self.tau else SystemRoute.SLOW

        metrics = {
            "complexity_score": complexity,
            "threshold_tau": self.tau,
            "correlation_strength": correlation_strength,
            "domain_crossing": domain_crossing,
            "stakeholder_count": float(stakeholder_count),
            "uncertainty": uncertainty,
        }
        return decision, complexity, metrics
