"""
Main Training & Simulation Script for the Neuromorphic Brain Architecture.
Orchestrates:
- Device selection (CUDA / CPU auto-detection).
- Dataset loading (Synthetic neuromorphic stream or N-MNIST).
- SNN training with DECOLLE local plasticity.
- Homeostatic state updates, CDR routing, memory logging, and sleep consolidation.
"""

import sys
import os
import time
import torch

# Ensure package path is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.dataset import get_neuromorphic_dataloader
from src.core.pipeline import NeuromorphicCognitivePipeline
from src.dashboard.monitor import global_monitor


def main():
    print("=" * 70)
    print("  🧠 NEUROMORPHIC BRAIN EMULATION (DECOLLE HYBRID ARCHITECTURE)")
    print("=" * 70)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Computing Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Configuration hyperparameters
    input_dim = 64
    time_steps = 16
    num_classes = 10
    epochs = 3
    batch_size = 16
    num_samples = 160

    print(f"[*] Initializing Neuromorphic Data Loader (Samples: {num_samples}, Steps: {time_steps})")
    loader = get_neuromorphic_dataloader(
        batch_size=batch_size,
        num_samples=num_samples,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
        data_dir="./data/n_mnist",
    )

    print("[*] Instantiating 6-Pillar Neuromorphic Cognitive Agent...")
    agent = NeuromorphicCognitivePipeline(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        num_classes=num_classes,
        device=device,
        lr=2e-3,
    )
    print("    [Pillar 1] DECOLLE Local Plasticity LIF Network: READY")
    print("    [Pillar 2] Hierarchical Memory & Sleep Engine:   READY")
    print("    [Pillar 3] Homeostatic State Vector Monitor:     READY")
    print("    [Pillar 4] Predictive Residual Error Module:     READY")
    print("    [Pillar 5] Cognitive Decision Router (CDR):      READY")
    print("    [Pillar 6] JEPA Latent World Model Simulator:    READY")
    print("-" * 70)

    # 3. Main Training & Perception Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n>>> Epoch {epoch}/{epochs} Active Cycle")
        epoch_acc = []
        epoch_surprises = []
        routes_taken = {"FAST_SYSTEM_1": 0, "SLOW_SYSTEM_2": 0}

        for batch_idx, (inputs, targets) in enumerate(loader):
            stats = agent.process_batch(inputs, targets, train_step=True)

            cur_v = stats["state_vector"]
            global_monitor.update({
                "curiosity": cur_v[0],
                "error_rate": cur_v[1],
                "resource_stress": cur_v[2],
                "uncertainty": cur_v[3],
                "behavioral_mode": stats["behavioral_mode"],
                "last_route": stats["route"],
                "last_accuracy": stats["accuracy"],
                "last_surprise": stats["surprise"],
                "episodic_events_count": len(agent.episodic_mem.retrieve_recent(1000)),
                "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                "log": f"Epoch {epoch} Batch {batch_idx+1}: Acc {stats['accuracy']*100:.1f}%, Route {stats['route']}",
            })

            epoch_acc.append(stats["accuracy"])
            epoch_surprises.append(stats["surprise"])
            routes_taken[stats["route"]] += 1

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                cur_v = stats["state_vector"]
                print(
                    f"  [Batch {batch_idx + 1:02d}/{len(loader):02d}] "
                    f"Acc: {stats['accuracy'] * 100:.1f}% | "
                    f"Surprise: {stats['surprise']:.4f} | "
                    f"Route: {stats['route']} | "
                    f"Mode: {stats['behavioral_mode']} | "
                    f"Homeostasis [C:{cur_v[0]:.2f}, E:{cur_v[1]:.2f}, R:{cur_v[2]:.2f}, U:{cur_v[3]:.2f}]"
                )

        mean_acc = sum(epoch_acc) / len(epoch_acc)
        mean_surp = sum(epoch_surprises) / len(epoch_surprises)
        print(f"--- Epoch {epoch} Summary: Mean Accuracy: {mean_acc * 100:.2f}%, Mean Surprise: {mean_surp:.4f}")
        print(f"--- Decision Routes: Fast (System 1): {routes_taken['FAST_SYSTEM_1']}, Slow (System 2): {routes_taken['SLOW_SYSTEM_2']}")

        # Homeostatic passive recovery
        agent.homeostasis.step_decay()

    # 4. Sleep Consolidation Phase
    print("\n" + "=" * 70)
    print("  🌙 ENTERING SLEEP CONSOLIDATION PHASE (NREM/REM Simulation)")
    print("=" * 70)
    consolidation_stats = agent.sleep_and_consolidate()
    print(f"[*] Episodic traces processed: {consolidation_stats['processed_events']}")
    print(f"[*] Concepts integrated into Semantic Graph: {consolidation_stats['consolidated_concepts']}")
    print(f"[*] Relational links established: {consolidation_stats['consolidated_relations']}")
    print(f"[*] Rogue traces rejected (Anti-Identity-Drift Guard): {consolidation_stats['rejected_for_drift']}")
    print(f"[*] Total Semantic Concepts in Graph: {agent.semantic_mem.num_concepts()}")
    print(f"[*] Total Semantic Relations in Graph: {agent.semantic_mem.num_relations()}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"✅ Cognitive cycle finished successfully in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
