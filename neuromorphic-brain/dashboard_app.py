"""
Interactive Dashboard and Monitoring Launcher.
Can run standalone or alongside the agent.
Supports:
1. --web: launches the live web dashboard on http://localhost:8000
2. --tui: renders a terminal-based live refreshing monitor
3. --simulate: generates a synthetic cognitive stream to visualize state shifts in real-time
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dashboard.monitor import global_monitor, start_dashboard_server
from src.core.dataset import get_neuromorphic_dataloader
from src.core.pipeline import NeuromorphicCognitivePipeline


def run_simulation(server_port: int, show_tui: bool = True):
    print(f"[*] Starting Neuromorphic Brain Web Dashboard on http://localhost:{server_port}")
    server = start_dashboard_server(host="0.0.0.0", port=server_port)

    print("[*] Instantiating Pipeline for Live Telemetry Stream...")
    agent = NeuromorphicCognitivePipeline(input_dim=64, hidden_dims=[128, 64], num_classes=10)
    loader = get_neuromorphic_dataloader(batch_size=16, num_samples=160)

    try:
        step = 0
        for epoch in range(1, 4):
            for batch_x, batch_y in loader:
                step += 1
                stats = agent.process_batch(batch_x, batch_y, train_step=True)
                sv = stats["state_vector"]

                # Update monitor
                global_monitor.update({
                    "curiosity": sv[0],
                    "error_rate": sv[1],
                    "resource_stress": sv[2],
                    "uncertainty": sv[3],
                    "behavioral_mode": stats["behavioral_mode"],
                    "last_route": stats["route"],
                    "last_accuracy": stats["accuracy"],
                    "last_surprise": stats["surprise"],
                    "episodic_events_count": len(agent.episodic_mem.retrieve_recent(1000)),
                    "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                    "log": f"Batch {step:02d}: Acc {stats['accuracy']*100:.1f}%, Route {stats['route']}, Surprise {stats['surprise']:.4f}",
                })

                if show_tui:
                    # Clear screen and render TUI
                    os.system("clear" if os.name == "posix" else "cls")
                    print(global_monitor.render_terminal_ascii())
                    print(f"\n🌐 Web UI available at: http://localhost:{server_port} (Press Ctrl+C to stop)")

                time.sleep(0.4)

            # Consolidate at end of epoch
            c_stats = agent.sleep_and_consolidate()
            global_monitor.update({
                "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                "log": f"Sleep cycle completed: {c_stats['consolidated_concepts']} concepts abstracted.",
            })
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping dashboard monitor.")
    finally:
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Brain Dashboard Monitor")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--no-tui", action="store_true", help="Disable terminal ASCII dashboard rendering")
    args = parser.parse_args()

    run_simulation(server_port=args.port, show_tui=not args.no_tui)


if __name__ == "__main__":
    main()
