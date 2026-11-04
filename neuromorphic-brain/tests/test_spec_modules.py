import pytest
import time
from src.core.synaptic_graph import SynapticGraphManager
from src.memory.hierarchy_controller import MemoryHierarchyController, SleepConsolidationWorker
from src.homeostasis.homeostatic_controller import GlobalStateVector, HomeostaticController
from src.predictive.predictive_unit import PredictiveProcessingUnit
from src.routing.dual_process_router import (
    System1FastEngine,
    ConfidenceEvaluator,
    System2SlowPlanner,
    DualProcessRouter,
)
from src.world_model.jepa_sandbox import JEPASandboxWorldModel


def test_synaptic_graph_hebbian_learning_and_pruning():
    sgm = SynapticGraphManager(learning_rate=0.2, decay_factor=0.05, initial_weight=0.3)
    sgm.add_node("neuron_A")
    sgm.add_node("neuron_B")

    # Before activation: no edge
    assert "neuron_A->neuron_B" not in sgm.edges

    now = time.time()
    sgm.trigger_activation("neuron_A", now)
    sgm.trigger_activation("neuron_B", now)

    # Co-activation within time window creates edge with initial weight
    sgm.update_local_learning("neuron_A", "neuron_B", time_window=1.0)
    assert "neuron_A->neuron_B" in sgm.edges
    assert sgm.edges["neuron_A->neuron_B"].weight == 0.3

    # Subsequent co-activation strengthens edge asymptotically
    sgm.trigger_activation("neuron_A", now)
    sgm.trigger_activation("neuron_B", now)
    sgm.update_local_learning("neuron_A", "neuron_B", time_window=1.0)
    # w_new = 0.3 + 0.2 * (1.0 - 0.3) = 0.44
    assert round(sgm.edges["neuron_A->neuron_B"].weight, 2) == 0.44

    # Pruning inactive edges: simulate stale edge
    sgm.edges["neuron_A->neuron_B"].last_activated = now - 20.0
    sgm.prune_inactive_edges(threshold=0.5, max_idle_time=5.0)
    assert "neuron_A->neuron_B" not in sgm.edges


def test_memory_hierarchy_and_sleep_consolidation():
    mhc = MemoryHierarchyController()
    mhc.write_working_memory("current_task", "compile_cortex")
    assert mhc.working_memory["current_task"] == "compile_cortex"

    mhc.store_episodic_event(
        event_type="learning_episode",
        payload={"subject": "NeuralNetwork", "predicate": "CognitiveModel", "detail": "LIF SNN"},
        vector_embedding=[0.1, 0.2, 0.3],
    )
    assert len(mhc.episodic_memory) == 1
    assert mhc.episodic_memory[0]["consolidated"] is False

    worker = SleepConsolidationWorker(memory_controller=mhc, cycle_threshold=50)
    assert worker.should_trigger_consolidation(current_cycle_count=50, is_idle=False) is True
    assert worker.should_trigger_consolidation(current_cycle_count=10, is_idle=False) is False
    assert worker.should_trigger_consolidation(current_cycle_count=10, is_idle=True) is True

    worker.run_sleep_consolidation()
    assert mhc.episodic_memory[0]["consolidated"] is True
    assert "NeuralNetwork" in mhc.semantic_graph
    assert "CognitiveModel" in mhc.semantic_graph["NeuralNetwork"]
    assert len(mhc.working_memory) == 0  # Working memory cleared after sleep


def test_homeostatic_controller_and_modulations():
    gsv = GlobalStateVector(curiosity=0.8, error_rate=0.05, energy_level=1.0, uncertainty=0.1)
    hc = HomeostaticController(state_vector=gsv)

    signals = hc.evaluate_and_modulate()
    assert signals["trigger_safety_protocol"] is False
    assert signals["recommended_system1_threshold"] == 0.80

    # High error rate triggers safety protocol and increases threshold
    hc.update_state(error_delta=0.3)
    signals_high_err = hc.evaluate_and_modulate()
    assert signals_high_err["trigger_safety_protocol"] is True
    assert signals_high_err["recommended_system1_threshold"] == 0.95

    # Low curiosity triggers exploration
    hc.update_state(curiosity_delta=-0.7)
    signals_low_cur = hc.evaluate_and_modulate()
    assert signals_low_cur["generate_exploration_task"] is True

    # Low energy reduces compute throughput
    hc.update_state(energy_delta=-0.9)
    signals_low_energy = hc.evaluate_and_modulate()
    assert signals_low_energy["reduce_compute_throughput"] is True


def test_predictive_processing_unit():
    ppu = PredictiveProcessingUnit(default_threshold=0.20)
    current = {"feature_x": 10.0, "feature_y": 5.0}

    pred = ppu.predict_next_state(current)
    assert pred["feature_x"] == 10.2
    assert pred["feature_y"] == 5.1

    # Minimal residual error
    actual_close = {"feature_x": 10.21, "feature_y": 5.09}
    err_low = ppu.calculate_residual_error(actual_close, pred)
    routed, msg = ppu.route_signal(err_low, threshold=0.20)
    assert routed is False
    assert "BYPASSED_UPPER_LAYERS" in msg

    # Large residual error (Surprise)
    actual_far = {"feature_x": 20.0, "feature_y": 15.0}
    err_high = ppu.calculate_residual_error(actual_far, pred)
    routed_high, msg_high = ppu.route_signal(err_high, threshold=0.20)
    assert routed_high is True
    assert "ROUTED_TO_UPPER_LAYERS" in msg_high


def test_dual_process_router_and_homeostatic_override():
    router = DualProcessRouter(default_threshold=0.80)

    # Routine task -> System 1 (confidence ~ 0.92 >= 0.80)
    res_fast = router.process_input("routine simple calculation")
    assert res_fast["execution_path"] == "SYSTEM_1_FAST"
    assert res_fast["confidence"] == 0.92

    # Complex task -> System 2 (confidence 0.60 < 0.80)
    res_slow = router.process_input("complex multi-step puzzle")
    assert res_slow["execution_path"] == "SYSTEM_2_SLOW"
    assert res_slow["confidence"] == 0.99

    # Dynamic threshold override from Homeostasis safety protocol (0.95)
    # Even simple task (0.92) is rerouted to System 2 when threshold is 0.95!
    res_overridden = router.process_input("routine simple calculation", dynamic_threshold_override=0.95)
    assert res_overridden["execution_path"] == "SYSTEM_2_SLOW"


def test_jepa_sandbox_world_model_and_rollout():
    jepa = JEPASandboxWorldModel()

    raw_state = {"velocity": 12.5, "light_intensity": 150}
    abstract_state = jepa.abstract_state_encoder(raw_state)
    assert "abstract_velocity" in abstract_state
    assert "abstract_light_intensity" in abstract_state

    # Valid candidate count: 5 to 10
    candidates = [
        "action_safe_step",
        "action_explore_turn",
        "action_wait",
        "action_scan_room",
        "action_risk_leap",
    ]
    outcomes = jepa.virtual_rollout(candidates, depth=5)
    assert len(outcomes) == 5

    best = jepa.evaluate_and_select_path(outcomes)
    assert "net_score" in best
    # Safe actions should have higher net_score than risk actions
    assert "risk" not in best["action"]

    # Candidate count validation error when < 5
    with pytest.raises(ValueError, match="szigorúan 5 és 10 közötti"):
        jepa.virtual_rollout(["act1", "act2"], depth=5)
