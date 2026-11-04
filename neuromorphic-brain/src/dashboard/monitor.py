"""
Dashboard Monitor Server & Terminal Visualizer.
Provides:
1. Live web API & single-page HTML/JS dashboard showing real-time gauges, logs, routes, and memory state.
2. Terminal text dashboard (TUI) with ASCII progress meters and real-time ANSI updates.
"""

from typing import Dict, Any, List, Optional
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse


class BrainStateMonitor:
    """Thread-safe state repository tracking agent state and telemetry."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state = {
            "curiosity": 0.5,
            "error_rate": 0.1,
            "resource_stress": 0.1,
            "uncertainty": 0.2,
            "behavioral_mode": "BALANCED",
            "last_route": "FAST_SYSTEM_1",
            "last_accuracy": 0.0,
            "last_surprise": 0.0,
            "working_memory_items": 0,
            "episodic_events_count": 0,
            "semantic_concepts_count": 0,
            "cycle_count": 0,
            "layer_activities": [
                {"layer_index": 1, "firing_rate": 0.0, "total_spikes": 0, "sample_neuron_activity": [0.0]*16},
                {"layer_index": 2, "firing_rate": 0.0, "total_spikes": 0, "sample_neuron_activity": [0.0]*16},
            ],
            "custom_input_queue": [],
            "recent_logs": [],
        }

    def update(self, payload: Dict[str, Any]):
        with self._lock:
            for k, v in payload.items():
                if k == "log":
                    timestamp = time.strftime("%H:%M:%S")
                    self.state["recent_logs"].insert(0, f"[{timestamp}] {v}")
                    self.state["recent_logs"] = self.state["recent_logs"][:15]
                elif k in self.state:
                    self.state[k] = v

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.state))

    def push_custom_training_input(self, target_class: int, firing_rate: float = 0.25, time_steps: int = 16) -> None:
        """Enqueues a user-specified training input to be processed by the live cognitive loop."""
        with self._lock:
            self.state["custom_input_queue"].append({
                "target_class": int(target_class),
                "firing_rate": float(firing_rate),
                "time_steps": int(time_steps),
                "timestamp": time.time(),
            })

    def pop_custom_training_input(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self.state["custom_input_queue"]:
                return self.state["custom_input_queue"].pop(0)
            return None

    def render_terminal_ascii(self) -> str:
        """Renders an ASCII text-based dashboard widget."""
        snap = self.get_snapshot()

        def bar(val: float, length: int = 20) -> str:
            filled = int(round(val * length))
            return "█" * filled + "░" * (length - filled)

        # SNN neuron activity raster visualization
        layer1_rate = snap["layer_activities"][0]["firing_rate"] if snap["layer_activities"] else 0.0
        layer2_rate = snap["layer_activities"][1]["firing_rate"] if len(snap["layer_activities"]) > 1 else 0.0

        def neuron_grid(activities: List[float]) -> str:
            chars = ["·", "░", "▒", "▓", "█"]
            out = ""
            for a in activities[:16]:
                idx = min(int(a * 4.9), 4)
                out += chars[idx] + " "
            return out.strip()

        l1_grid = neuron_grid(snap["layer_activities"][0].get("sample_neuron_activity", [])) if snap["layer_activities"] else ""
        l2_grid = neuron_grid(snap["layer_activities"][1].get("sample_neuron_activity", [])) if len(snap["layer_activities"]) > 1 else ""

        lines = [
            "╔══════════════════════════════════════════════════════════════════════════╗",
            "║                  🧠 NEUROMORPHIC BRAIN MONITOR TUI                      ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║ Mode: {snap['behavioral_mode']:<14} | Last Decision Route: {snap['last_route']:<18}      ║",
            f"║ Cycle: {snap['cycle_count']:<13} | Accuracy: {snap['last_accuracy']*100:>5.1f}% | Surprise: {snap['last_surprise']:>6.4f} ║",
            "╟──────────────────────────────────────────────────────────────────────────╢",
            f"║ Curiosity:       [{bar(snap['curiosity'])}] {snap['curiosity']:>4.2f}                   ║",
            f"║ Error Rate:      [{bar(snap['error_rate'])}] {snap['error_rate']:>4.2f}                   ║",
            f"║ Resource Stress: [{bar(snap['resource_stress'])}] {snap['resource_stress']:>4.2f}                   ║",
            f"║ Uncertainty:     [{bar(snap['uncertainty'])}] {snap['uncertainty']:>4.2f}                   ║",
            "╟─────────────────── ⚡ LIVE SNN NEURON NETWORK ACTIVITY ─────────────────╢",
            f"║ Layer 1 (Spikes): Rate {layer1_rate*100:>4.1f}% [{l1_grid:<31}] ║",
            f"║ Layer 2 (Spikes): Rate {layer2_rate*100:>4.1f}% [{l2_grid:<31}] ║",
            "╟──────────────────────────────────────────────────────────────────────────╢",
            f"║ Episodic Events: {snap['episodic_events_count']:<5} | Semantic Concepts: {snap['semantic_concepts_count']:<5}                    ║",
            "╟────────────────────────────── RECENT LOGS ───────────────────────────────╢",
        ]
        logs = snap["recent_logs"][:4]
        if not logs:
            lines.append("║ (No logs recorded yet)                                                   ║")
        else:
            for l in logs:
                lines.append(f"║ {l[:72]:<72} ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# Singleton monitor instance
global_monitor = BrainStateMonitor()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Neuromorphic Brain Emulation Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --accent: #38bdf8;
            --fast: #10b981;
            --slow: #f59e0b;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #334155;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }
        h1 { margin: 0; font-size: 1.6rem; display: flex; align-items: center; gap: 10px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        .card {
            background-color: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            border: 1px solid #334155;
        }
        .card h2 { margin-top: 0; font-size: 1.1rem; color: var(--accent); }
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .bar-container {
            width: 100%;
            height: 10px;
            background: #334155;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 4px;
        }
        .bar {
            height: 100%;
            background: var(--accent);
            width: 0%;
            transition: width 0.3s ease;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 0.85rem;
        }
        .route-fast { background-color: #064e3b; color: #6ee7b7; border: 1px solid var(--fast); }
        .route-slow { background-color: #78350f; color: #fde68a; border: 1px solid var(--slow); }
        .logs-box {
            background: #020617;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.85rem;
            height: 160px;
            overflow-y: auto;
            border: 1px solid #1e293b;
        }
        .log-entry { margin-bottom: 5px; color: #cbd5e1; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Neuromorphic Brain Live Dashboard</h1>
        <div><span id="badge-mode" class="badge route-fast">BALANCED</span></div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Homeostatic State Vector</h2>
            <div class="metric-row">
                <span>Curiosity:</span>
                <span id="val-curiosity">0.00</span>
            </div>
            <div class="bar-container"><div id="bar-curiosity" class="bar"></div></div>
            <br>
            <div class="metric-row">
                <span>Error Rate:</span>
                <span id="val-error">0.00</span>
            </div>
            <div class="bar-container"><div id="bar-error" class="bar" style="background:#ef4444;"></div></div>
            <br>
            <div class="metric-row">
                <span>Resource Stress:</span>
                <span id="val-stress">0.00</span>
            </div>
            <div class="bar-container"><div id="bar-stress" class="bar" style="background:#f97316;"></div></div>
            <br>
            <div class="metric-row">
                <span>Uncertainty:</span>
                <span id="val-uncertainty">0.00</span>
            </div>
            <div class="bar-container"><div id="bar-uncertainty" class="bar" style="background:#a855f7;"></div></div>
        </div>

        <div class="card">
            <h2>Cognitive Routing & Performance</h2>
            <div class="metric-row">
                <span>Decision Route:</span>
                <span id="badge-route" class="badge route-fast">FAST_SYSTEM_1</span>
            </div>
            <div class="metric-row" style="margin-top: 15px;">
                <span>Accuracy:</span>
                <strong id="val-acc" style="font-size: 1.2rem; color: #34d399;">0.0%</strong>
            </div>
            <div class="metric-row">
                <span>Surprise (Residual Error):</span>
                <strong id="val-surprise" style="font-size: 1.1rem; color: #38bdf8;">0.0000</strong>
            </div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 15px 0;">
            <div class="metric-row">
                <span>Episodic Events:</span>
                <strong id="val-episodic">0</strong>
            </div>
            <div class="metric-row">
                <span>Semantic Concepts:</span>
                <strong id="val-semantic">0</strong>
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>⚡ Live SNN Neuron Network Visualizer</h2>
            <div class="metric-row">
                <span>Layer 1 Firing Rate:</span>
                <span id="val-l1-rate" style="color: #38bdf8; font-weight: bold;">0.0%</span>
            </div>
            <div class="bar-container"><div id="bar-l1-rate" class="bar"></div></div>
            <div style="margin-top: 10px; font-size: 0.85rem; color: #94a3b8;">Layer 1 Sample Neurons Activity:</div>
            <div id="grid-l1" style="display: flex; gap: 4px; margin-top: 5px; flex-wrap: wrap;"></div>

            <br>
            <div class="metric-row">
                <span>Layer 2 Firing Rate:</span>
                <span id="val-l2-rate" style="color: #a855f7; font-weight: bold;">0.0%</span>
            </div>
            <div class="bar-container"><div id="bar-l2-rate" class="bar" style="background: #a855f7;"></div></div>
            <div style="margin-top: 10px; font-size: 0.85rem; color: #94a3b8;">Layer 2 Sample Neurons Activity:</div>
            <div id="grid-l2" style="display: flex; gap: 4px; margin-top: 5px; flex-wrap: wrap;"></div>
        </div>

        <div class="card">
            <h2>🎯 Inject Live Training Input</h2>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0;">
                Send custom neuromorphic spike pattern directly into the active learning loop in real-time.
            </p>
            <div style="margin-bottom: 12px;">
                <label style="display: block; font-size: 0.85rem; margin-bottom: 4px;">Target Digit / Concept Class (0-9):</label>
                <input id="input-class" type="number" min="0" max="9" value="3" style="width: 100%; padding: 8px; border-radius: 6px; background: #0f172a; color: white; border: 1px solid #334155;">
            </div>
            <div style="margin-bottom: 12px;">
                <label style="display: block; font-size: 0.85rem; margin-bottom: 4px;">Spike Firing Intensity (0.05 - 0.8):</label>
                <input id="input-rate" type="range" min="0.05" max="0.8" step="0.05" value="0.3" style="width: 100%;" oninput="document.getElementById('rate-val').innerText = this.value">
                <span id="rate-val" style="font-size: 0.85rem; color: #38bdf8;">0.3</span>
            </div>
            <button onclick="injectInput()" style="background: #2563eb; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; width: 100%;">
                ⚡ Inject Spike Stream into Brain
            </button>
            <div id="inject-status" style="margin-top: 8px; font-size: 0.85rem; color: #34d399;"></div>
        </div>
    </div>

    <div class="card">
        <h2>Live System Telemetry Logs</h2>
        <div id="logs-box" class="logs-box">
            <div class="log-entry">Waiting for agent loop to report telemetry...</div>
        </div>
    </div>

    <script>
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();

                // State vector
                document.getElementById('val-curiosity').innerText = data.curiosity.toFixed(2);
                document.getElementById('bar-curiosity').style.width = (data.curiosity * 100) + '%';

                document.getElementById('val-error').innerText = data.error_rate.toFixed(2);
                document.getElementById('bar-error').style.width = (data.error_rate * 100) + '%';

                document.getElementById('val-stress').innerText = data.resource_stress.toFixed(2);
                document.getElementById('bar-stress').style.width = (data.resource_stress * 100) + '%';

                document.getElementById('val-uncertainty').innerText = data.uncertainty.toFixed(2);
                document.getElementById('bar-uncertainty').style.width = (data.uncertainty * 100) + '%';

                // Mode & Route
                const modeEl = document.getElementById('badge-mode');
                modeEl.innerText = data.behavioral_mode;

                const routeEl = document.getElementById('badge-route');
                routeEl.innerText = data.last_route;
                if (data.last_route.includes('SLOW')) {
                    routeEl.className = 'badge route-slow';
                } else {
                    routeEl.className = 'badge route-fast';
                }

                document.getElementById('val-acc').innerText = (data.last_accuracy * 100).toFixed(1) + '%';
                document.getElementById('val-surprise').innerText = data.last_surprise.toFixed(4);
                document.getElementById('val-episodic').innerText = data.episodic_events_count;
                document.getElementById('val-semantic').innerText = data.semantic_concepts_count;

                // Neuron visualizer grids
                if (data.layer_activities && data.layer_activities.length >= 2) {
                    const l1 = data.layer_activities[0];
                    const l2 = data.layer_activities[1];

                    document.getElementById('val-l1-rate').innerText = (l1.firing_rate * 100).toFixed(1) + '%';
                    document.getElementById('bar-l1-rate').style.width = Math.min(100, l1.firing_rate * 300) + '%';

                    document.getElementById('val-l2-rate').innerText = (l2.firing_rate * 100).toFixed(1) + '%';
                    document.getElementById('bar-l2-rate').style.width = Math.min(100, l2.firing_rate * 300) + '%';

                    const renderCells = (acts, color) => acts.map(a => {
                        const alpha = Math.max(0.15, Math.min(1.0, a * 2.5));
                        return `<div style="width: 14px; height: 14px; border-radius: 3px; background: ${color}; opacity: ${alpha}; border: 1px solid rgba(255,255,255,0.1);" title="Activity: ${a}"></div>`;
                    }).join('');

                    document.getElementById('grid-l1').innerHTML = renderCells(l1.sample_neuron_activity || [], '#38bdf8');
                    document.getElementById('grid-l2').innerHTML = renderCells(l2.sample_neuron_activity || [], '#a855f7');
                }

                // Logs
                if (data.recent_logs && data.recent_logs.length > 0) {
                    const logsBox = document.getElementById('logs-box');
                    logsBox.innerHTML = data.recent_logs.map(l => `<div class="log-entry">${l}</div>`).join('');
                }
            } catch(e) {
                console.error("Fetch error", e);
            }
        }

        async function injectInput() {
            const targetClass = parseInt(document.getElementById('input-class').value);
            const firingRate = parseFloat(document.getElementById('input-rate').value);
            const statusEl = document.getElementById('inject-status');
            try {
                const res = await fetch('/api/inject_input', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ target_class: targetClass, firing_rate: firingRate, time_steps: 16 })
                });
                const resData = await res.json();
                statusEl.innerText = `✓ Spike pattern for Class ${targetClass} enqueued!`;
                setTimeout(() => statusEl.innerText = '', 3000);
            } catch (err) {
                statusEl.innerText = `Error: ${err}`;
            }
        }

        setInterval(fetchState, 500);
        fetchState();
    </script>
</body>
</html>
"""


class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    """Simple standard library HTTP handler serving HTML and JSON state endpoints."""

    def log_message(self, format, *args):
        # Silence default standard HTTP access logs
        return

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path
        if url_path == "/" or url_path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif url_path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            snapshot = global_monitor.get_snapshot()
            self.wfile.write(json.dumps(snapshot).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url_path = urllib.parse.urlparse(self.path).path
        if url_path == "/api/update":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                global_monitor.update(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
        elif url_path == "/api/inject_input":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                target_class = int(data.get("target_class", 0))
                firing_rate = float(data.get("firing_rate", 0.3))
                time_steps = int(data.get("time_steps", 16))
                global_monitor.push_custom_training_input(target_class, firing_rate, time_steps)
                global_monitor.update({"log": f"User injected custom training input for Class {target_class}"})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "queued", "target_class": target_class}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def start_dashboard_server(host: str = "0.0.0.0", port: int = 8000) -> HTTPServer:
    """Starts the dashboard HTTP server in a background daemon thread."""
    server = HTTPServer((host, port), DashboardHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
