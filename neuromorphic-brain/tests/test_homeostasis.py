import pytest
from src.homeostasis.state_vector import HomeostaticState, BehavioralMode


def test_homeostatic_initialization_and_vector():
    hs = HomeostaticState(curiosity=0.7, error_rate=0.2, resource_stress=0.3, uncertainty=0.4)
    vec = hs.vector
    assert len(vec) == 4
    assert vec == [0.7, 0.2, 0.3, 0.4]


def test_homeostatic_behavioral_modes():
    # Exploration mode: high curiosity, low error
    h_explore = HomeostaticState(curiosity=0.85, error_rate=0.1, resource_stress=0.1, uncertainty=0.2)
    assert h_explore.get_behavioral_mode() == BehavioralMode.EXPLORATION

    # Cautious mode: high error rate
    h_cautious_err = HomeostaticState(curiosity=0.8, error_rate=0.6, resource_stress=0.2, uncertainty=0.1)
    assert h_cautious_err.get_behavioral_mode() == BehavioralMode.CAUTIOUS

    # Cautious mode: high uncertainty
    h_cautious_unc = HomeostaticState(curiosity=0.2, error_rate=0.1, resource_stress=0.2, uncertainty=0.7)
    assert h_cautious_unc.get_behavioral_mode() == BehavioralMode.CAUTIOUS

    # Conservation mode: high resource stress
    h_stress = HomeostaticState(curiosity=0.9, error_rate=0.9, resource_stress=0.85, uncertainty=0.9)
    assert h_stress.get_behavioral_mode() == BehavioralMode.CONSERVATION


def test_homeostatic_update_and_decay():
    hs = HomeostaticState(curiosity=0.2, error_rate=0.1, resource_stress=0.1, uncertainty=0.1)
    # High surprise should increase curiosity
    hs.update_observation(surprise=0.9, task_error=0.0, system_load=0.0, entropy=0.1, smoothing=0.5)
    assert hs.curiosity > 0.4

    # High task error should increase error rate
    hs.update_observation(surprise=0.0, task_error=0.8, system_load=0.0, entropy=0.0, smoothing=0.5)
    assert hs.error_rate > 0.3

    # Step decay reduces levels towards resting baseline
    prev_curiosity = hs.curiosity
    hs.step_decay()
    assert hs.curiosity < prev_curiosity
