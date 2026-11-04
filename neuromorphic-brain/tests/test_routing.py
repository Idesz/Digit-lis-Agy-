import pytest
from src.routing.cdr_router import CDRRouter, SystemRoute


def test_cdr_routing_fast_path():
    router = CDRRouter(tau=0.5)
    # Familiar query: high correlation, low domain crossing, low stakeholders, low uncertainty
    decision, score, metrics = router.route(
        correlation_strength=0.95,
        domain_crossing=0.1,
        stakeholder_count=1,
        uncertainty=0.1,
    )
    assert decision == SystemRoute.FAST
    assert score < router.tau
    assert metrics["complexity_score"] == score


def test_cdr_routing_slow_path():
    router = CDRRouter(tau=0.5)
    # Complex query: unfamiliar, high domain crossing, many stakeholders, high uncertainty
    decision, score, metrics = router.route(
        correlation_strength=0.1,
        domain_crossing=0.9,
        stakeholder_count=8,
        uncertainty=0.85,
    )
    assert decision == SystemRoute.SLOW
    assert score >= router.tau


def test_cdr_weight_normalization():
    router = CDRRouter(
        tau=0.4,
        w_correlation=1.0,
        w_domain=1.0,
        w_stakeholders=1.0,
        w_uncertainty=1.0,
    )
    total = router.w_correlation + router.w_domain + router.w_stakeholders + router.w_uncertainty
    assert abs(total - 1.0) < 1e-5
