import pytest
import urllib.request
import json
from src.dashboard.monitor import BrainStateMonitor, global_monitor, start_dashboard_server


def test_brain_state_monitor_updates_and_ascii_render():
    monitor = BrainStateMonitor()
    monitor.update({
        "curiosity": 0.85,
        "error_rate": 0.12,
        "resource_stress": 0.35,
        "uncertainty": 0.22,
        "behavioral_mode": "EXPLORATION",
        "last_route": "FAST_SYSTEM_1",
        "last_accuracy": 0.92,
        "last_surprise": 0.045,
        "episodic_events_count": 14,
        "semantic_concepts_count": 6,
        "log": "Test cognitive event executed successfully.",
    })

    snap = monitor.get_snapshot()
    assert snap["curiosity"] == 0.85
    assert snap["behavioral_mode"] == "EXPLORATION"
    assert snap["last_accuracy"] == 0.92
    assert len(snap["recent_logs"]) == 1

    # Check ASCII render
    rendered = monitor.render_terminal_ascii()
    assert "NEUROMORPHIC BRAIN MONITOR" in rendered
    assert "EXPLORATION" in rendered
    assert "FAST_SYSTEM_1" in rendered


def test_dashboard_server_endpoints():
    server = start_dashboard_server(host="127.0.0.1", port=8989)
    try:
        # GET / (HTML)
        req = urllib.request.Request("http://127.0.0.1:8989/")
        with urllib.request.urlopen(req) as response:
            assert response.status == 200
            html = response.read().decode("utf-8")
            assert "Neuromorphic Brain Live Dashboard" in html

        # GET /api/state (JSON)
        req_api = urllib.request.Request("http://127.0.0.1:8989/api/state")
        with urllib.request.urlopen(req_api) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert "curiosity" in data
            assert "behavioral_mode" in data

        # POST /api/update
        update_payload = json.dumps({"curiosity": 0.99, "log": "External telemetry update"}).encode("utf-8")
        req_post = urllib.request.Request(
            "http://127.0.0.1:8989/api/update",
            data=update_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_post) as response:
            assert response.status == 200
            res_json = json.loads(response.read().decode("utf-8"))
            assert res_json["status"] == "ok"

        # POST /api/inject_input (Custom Training Input Injection)
        inject_payload = json.dumps({"target_class": 7, "firing_rate": 0.45, "time_steps": 16}).encode("utf-8")
        req_inject = urllib.request.Request(
            "http://127.0.0.1:8989/api/inject_input",
            data=inject_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_inject) as response:
            assert response.status == 200
            res_inj = json.loads(response.read().decode("utf-8"))
            assert res_inj["status"] == "queued"
            assert res_inj["target_class"] == 7

        queued_item = global_monitor.pop_custom_training_input()
        assert queued_item is not None
        assert queued_item["target_class"] == 7
        assert queued_item["firing_rate"] == 0.45

    finally:
        server.shutdown()
