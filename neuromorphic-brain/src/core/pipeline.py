"""
Orchestrator for the Complete Neuromorphic Cognitive Architecture.
Integrates:
1. DECOLLE Local Plasticity SNN
2. Multi-level Memory (Working, Episodic, Semantic, Sleep Consolidation)
3. Global State Vector & Homeostasis
4. Residual Error Predictive Processing
5. Cognitive Decision Routing (CDR System 1 / System 2)
6. JEPA World Model Simulation
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.decolle_layer import DECOLLENetwork
from src.memory.working_memory import WorkingMemory
from src.memory.episodic_memory import EpisodicMemory, EpisodicEvent
from src.memory.semantic_memory import SemanticMemory
from src.memory.sleep_consolidation import SleepConsolidator
from src.homeostasis.state_vector import HomeostaticState, BehavioralMode
from src.predictive.residual_error import ResidualPredictiveModule
from src.routing.cdr_router import CDRRouter, SystemRoute
from src.world_model.jepa_simulator import JEPASimulator


class NeuromorphicCognitivePipeline:
    """
    End-to-end cognitive agent orchestrator executing perception, local learning,
    routing, memory consolidation, and world-model simulations.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list[int] = None,
        num_classes: int = 10,
        device: torch.device = None,
        lr: float = 1e-3,
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Pillar 1: SNN with DECOLLE local plasticity
        all_layer_dims = [input_dim] + hidden_dims
        self.snn = DECOLLENetwork(layer_sizes=all_layer_dims, num_classes=num_classes).to(self.device)
        self.optimizers = self.snn.get_adamax_optimizers(lr=lr)

        # Pillar 2: Memory hierarchy
        self.working_mem = WorkingMemory()
        self.episodic_mem = EpisodicMemory(vector_dim=hidden_dims[-1])
        self.semantic_mem = SemanticMemory()
        self.sleep_worker = SleepConsolidator(
            episodic_memory=self.episodic_mem,
            semantic_memory=self.semantic_mem,
        )

        # Pillar 3: Homeostasis & State Vector
        self.homeostasis = HomeostaticState()

        # Pillar 4: Predictive Residual Coding
        self.predictive = ResidualPredictiveModule(input_dim=input_dim, latent_dim=hidden_dims[0]).to(self.device)
        self.pred_optimizer = torch.optim.Adam(self.predictive.parameters(), lr=1e-3)

        # Pillar 5: Cognitive Decision Router (CDR)
        self.router = CDRRouter(tau=0.5)

        # Pillar 6: JEPA World Model
        self.world_model = JEPASimulator(
            latent_dim=hidden_dims[-1],
            action_dim=num_classes,
            hidden_dim=128,
        ).to(self.device)
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)

    def process_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        train_step: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes a complete cognitive cycle over a batch of spatio-temporal inputs:
        inputs: [batch_size, time_steps, input_dim]
        targets: [batch_size]
        """
        batch_size, time_steps, _ = inputs.shape
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # 1. Predictive Coding & Surprise Evaluation
        # Evaluate mean input pattern over time
        mean_input = inputs.mean(dim=1)  # [batch_size, input_dim]
        pred_loss, mean_surprise, is_plastic = self.predictive.gated_loss(mean_input)

        if train_step and is_plastic:
            self.pred_optimizer.zero_grad()
            pred_loss.backward()
            self.pred_optimizer.step()

        # 2. DECOLLE SNN Local Learning
        layer_spikes, layer_readouts = self.snn.forward_sequence(inputs)

        layer_losses = []
        if train_step:
            for layer, spikes, readout, opt in zip(self.snn.layers, layer_spikes, layer_readouts, self.optimizers):
                opt.zero_grad()
                loss = layer.compute_local_loss(readout, targets, spikes, time_steps)
                loss.backward()
                opt.step()
                layer_losses.append(loss.item())

        # Top layer outputs
        final_readout = layer_readouts[-1]
        preds = final_readout.argmax(dim=-1)
        batch_accuracy = (preds == targets).float().mean().item()
        task_error = 1.0 - batch_accuracy

        # Latent representation from top SNN layer (mean spike trace)
        top_spikes = layer_spikes[-1].mean(dim=1)  # [batch_size, hidden_dims[-1]]
        latent_repr = top_spikes[0].detach().cpu().numpy().tolist()

        # 3. CDR Decision Routing
        # Retrieve similar past memories to compute familiarity / correlation
        similar_episodes = self.episodic_mem.search_similar(latent_repr, limit=1)
        corr_strength = similar_episodes[0][1] if similar_episodes else 0.2

        decision, complexity, metrics = self.router.route(
            correlation_strength=float(corr_strength),
            domain_crossing=0.2 if batch_accuracy > 0.5 else 0.7,
            stakeholder_count=1,
            uncertainty=float(self.homeostasis.uncertainty),
        )

        # 4. If System 2 (Slow), invoke JEPA World Model Virtual Rollout
        simulated_rollout_val = None
        if decision == SystemRoute.SLOW:
            s_0 = top_spikes[:1].detach()  # first sample [1, latent_dim] (detached latent state for world model)
            # Mock candidate action sequence: one-hot for candidate classes
            actions = torch.zeros(1, 3, self.num_classes, device=self.device)
            actions[0, :, targets[0]] = 1.0
            traj, total_val = self.world_model.simulate_rollout(s_0, actions)
            simulated_rollout_val = total_val.item()

            if train_step:
                self.wm_optimizer.zero_grad()
                wm_loss = F.mse_loss(traj[-1], s_0)
                wm_loss.backward()
                self.wm_optimizer.step()

        # 5. Working Memory & Episodic Logging
        self.working_mem.put(
            key="current_perceptual_state",
            data={"predicted_class": int(preds[0].item()), "complexity": complexity},
            ttl=300,
        )

        # Store in episodic memory
        event = EpisodicEvent(
            embedding=latent_repr,
            context={
                "concept": f"class_{targets[0].item()}",
                "category": "digit",
                "route": decision.value,
            },
            surprise=float(mean_surprise.item()),
        )
        self.episodic_mem.store_event(event)

        # 6. Update Homeostatic State Vector
        self.homeostasis.update_observation(
            surprise=float(mean_surprise.item()),
            task_error=task_error,
            system_load=0.15,
            entropy=float(complexity),
        )
        mode = self.homeostasis.get_behavioral_mode()

        # Compute layer-level spiking network activity for live visualization
        layer_activities = []
        for l_idx, l_spikes in enumerate(layer_spikes):
            mean_firing_rate = l_spikes.mean().item()
            total_spikes = int(l_spikes.sum().item())
            # Sample of neuron states (first 16 neurons mean activity)
            sample_neurons = [round(float(v), 2) for v in l_spikes[0].mean(dim=0)[:16].tolist()]
            layer_activities.append({
                "layer_index": l_idx + 1,
                "firing_rate": mean_firing_rate,
                "total_spikes": total_spikes,
                "sample_neuron_activity": sample_neurons,
            })

        return {
            "accuracy": batch_accuracy,
            "pred_loss": pred_loss.item(),
            "surprise": mean_surprise.item(),
            "layer_losses": layer_losses,
            "route": decision.value,
            "complexity": complexity,
            "behavioral_mode": mode.value,
            "state_vector": self.homeostasis.vector,
            "simulated_rollout_val": simulated_rollout_val,
            "layer_activities": layer_activities,
            "predicted_class": int(preds[0].item()),
            "target_class": int(targets[0].item()),
        }

    def sleep_and_consolidate(self) -> Dict[str, int]:
        """Runs offline sleep consolidation cycle."""
        return self.sleep_worker.consolidate()
