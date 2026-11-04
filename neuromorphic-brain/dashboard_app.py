"""
Interactive Dashboard and Monitoring Launcher.
Can run standalone or alongside the agent.
Supports:
1. --web: launches the live web dashboard on http://localhost:8000
2. --tui: renders a terminal-based live refreshing monitor
3. --continuous: runs continuous, uninterrupted real-time cognitive cycles
4. Live training input injection: dynamically receives custom spike patterns via Web UI / API
"""

import sys
import os
import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dashboard.monitor import global_monitor, start_dashboard_server
from src.core.dataset import get_neuromorphic_dataloader
from src.core.pipeline import NeuromorphicCognitivePipeline


def generate_custom_spike_batch(
    target_class: int,
    firing_rate: float = 0.3,
    time_steps: int = 16,
    input_dim: int = 64,
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a synthetic spike batch tailored to a specific class for interactive training."""
    labels = torch.full((batch_size,), target_class, dtype=torch.long)
    samples = torch.zeros(batch_size, time_steps, input_dim)

    start_idx = (target_class * (input_dim // 10)) % input_dim
    end_idx = min(start_idx + (input_dim // 10), input_dim)

    for i in range(batch_size):
        noise = (torch.rand(time_steps, input_dim) < (firing_rate * 0.2)).float()
        signal = (torch.rand(time_steps, end_idx - start_idx) < (firing_rate * 2.5)).float()
        samples[i] = noise
        samples[i, :, start_idx:end_idx] = signal

    return samples, labels


def run_continuous_pipeline(server_port: int, show_tui: bool = True, continuous: bool = False, max_epochs: int = 3):
    print(f"[*] Starting Neuromorphic Brain Web Dashboard on http://localhost:{server_port}")
    server = start_dashboard_server(host="0.0.0.0", port=server_port)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Instantiating Neuromorphic Cognitive Agent on {device.type.upper()}...")
    agent = NeuromorphicCognitivePipeline(input_dim=64, hidden_dims=[128, 64], num_classes=10, device=device)
    loader = get_neuromorphic_dataloader(batch_size=16, num_samples=160)

    try:
        step = 0
        epoch = 0
        while True:
            epoch += 1
            if not continuous and epoch > max_epochs:
                print("\n[✓] Specified training epochs completed. Remaining in interactive live listening mode...")
                # Stay in live monitoring mode to continue responding to user UI actions
                while True:
                    custom_input = global_monitor.pop_custom_training_input()
                    if custom_input:
                        step += 1
                        inputs, targets = generate_custom_spike_batch(
                            target_class=custom_input["target_class"],
                            firing_rate=custom_input["firing_rate"],
                            time_steps=custom_input["time_steps"],
                        )
                        stats = agent.process_batch(inputs, targets, train_step=True)
                        sv = stats["state_vector"]
                        global_monitor.update({
                            "curiosity": sv[0],
                            "error_rate": sv[1],
                            "resource_stress": sv[2],
                            "uncertainty": sv[3],
                            "cycle_count": step,
                            "behavioral_mode": stats["behavioral_mode"],
                            "last_route": stats["route"],
                            "last_accuracy": stats["accuracy"],
                            "last_surprise": stats["surprise"],
                            "layer_activities": stats.get("layer_activities", []),
                            "log": f"⚡ INJECTED Input (Class {custom_input['target_class']}): Acc {stats['accuracy']*100:.1f}%, Route {stats['route']}",
                        })
                    if show_tui:
                        os.system("clear" if os.name == "posix" else "cls")
                        print(global_monitor.render_terminal_ascii())
                        print(f"\n🌐 Web UI: http://localhost:{server_port} | Interactive Mode (Ctrl+C to stop)")
                    time.sleep(0.5)

            for batch_x, batch_y in loader:
                step += 1

                # Check if user injected custom training input via UI
                custom_input = global_monitor.pop_custom_training_input()
                if custom_input:
                    batch_x, batch_y = generate_custom_spike_batch(
                        target_class=custom_input["target_class"],
                        firing_rate=custom_input["firing_rate"],
                        time_steps=custom_input["time_steps"],
                    )
                    prefix = f"⚡ INJECTED (Class {custom_input['target_class']})"
                else:
                    prefix = f"Cycle {step:02d}"

                stats = agent.process_batch(batch_x, batch_y, train_step=True)
                sv = stats["state_vector"]

                global_monitor.update({
                    "curiosity": sv[0],
                    "error_rate": sv[1],
                    "resource_stress": sv[2],
                    "uncertainty": sv[3],
                    "cycle_count": step,
                    "behavioral_mode": stats["behavioral_mode"],
                    "last_route": stats["route"],
                    "last_accuracy": stats["accuracy"],
                    "last_surprise": stats["surprise"],
                    "episodic_events_count": len(agent.episodic_mem.retrieve_recent(1000)),
                    "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                    "layer_activities": stats.get("layer_activities", []),
                    "log": f"{prefix}: Acc {stats['accuracy']*100:.1f}%, Route {stats['route']}, Surprise {stats['surprise']:.4f}",
                })

                if show_tui:
                    os.system("clear" if os.name == "posix" else "cls")
                    print(global_monitor.render_terminal_ascii())
                    print(f"\n🌐 Web UI available at: http://localhost:{server_port} (Press Ctrl+C to stop)")

                time.sleep(0.3)

            # Consolidate memory at periodic intervals
            c_stats = agent.sleep_and_consolidate()
            global_monitor.update({
                "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                "log": f"Sleep consolidation: {c_stats['consolidated_concepts']} concepts abstracted into knowledge graph.",
            })
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping dashboard monitor.")
    finally:
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Brain Dashboard Monitor")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--no-tui", action="store_true", help="Disable terminal ASCII dashboard rendering")
    parser.add_argument("--continuous", action="store_true", help="Run uninterrupted continuous live learning loop")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs if not continuous")
    args = parser.parse_args()

    run_continuous_pipeline(server_port=args.port, show_tui=not args.no_tui, continuous=args.continuous, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
