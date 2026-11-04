"""
DECOLLE (Deep Continuous Local Learning) Spiking Neural Network Layer.
Implements:
- Leaky Integrate-and-Fire (LIF) neuron dynamics.
- Smooth surrogate gradient for non-differentiable spike activation.
- Local random readout mechanism to compute local classification error without global BPTT.
- AdaMax optimizer configuration (beta1=0, beta2=0.95).
- Activity (firing rate) and activation regularizations.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast sigmoid surrogate gradient for non-differentiable Heaviside step function.
    Forward: Heaviside step function Theta(x - threshold).
    Backward: Derivative of fast sigmoid 1 / (1 + gamma * |x - threshold|)^2.
    """

    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, threshold: float = 1.0, gamma: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.gamma = gamma
        spikes = (membrane_potential >= threshold).float()
        return spikes

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (membrane_potential,) = ctx.saved_tensors
        threshold = ctx.threshold
        gamma = ctx.gamma
        diff = torch.abs(membrane_potential - threshold)
        grad_surrogate = 1.0 / (1.0 + gamma * diff).pow(2)
        grad_input = grad_output * grad_surrogate
        return grad_input, None, None


surrogate_spike = FastSigmoidSurrogate.apply


class DECOLLELayer(nn.Module):
    """
    Individual DECOLLE layer featuring local synaptic plasticity.
    Each layer owns a local random readout that computes local errors,
    preventing BPTT gradient propagation to prior layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_classes: int = 10,
        tau_m: float = 20.0,
        dt: float = 1.0,
        v_th: float = 1.0,
        gamma: float = 10.0,
        trace_decay: float = 0.95,
        target_rate: float = 0.05,
        reg_rate_coef: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.v_th = v_th
        self.gamma = gamma
        self.alpha = math.exp(-dt / tau_m)
        self.trace_decay = trace_decay
        self.target_rate = target_rate
        self.reg_rate_coef = reg_rate_coef

        # Synaptic weights
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

        # Fixed random readout for local error generation (DECOLLE random readout)
        self.random_readout = nn.Linear(out_features, num_classes, bias=False)
        nn.init.normal_(self.random_readout.weight, mean=0.0, std=1.0 / math.sqrt(out_features))
        self.random_readout.weight.requires_grad = False

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes membrane potential U and smoothed spike trace S_bar.
        """
        u = torch.zeros(batch_size, self.out_features, device=device)
        s_bar = torch.zeros(batch_size, self.out_features, device=device)
        return u, s_bar

    def forward_step(
        self,
        input_spikes: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Processes a single time-step dt.
        input_spikes: [batch_size, in_features]
        Returns:
            spikes: [batch_size, out_features]
            next_state: (next_u, next_s_bar)
            readout_logits: [batch_size, num_classes] (local linear readout)
        """
        batch_size = input_spikes.size(0)
        device = input_spikes.device

        if state is None:
            u, s_bar = self.init_state(batch_size, device)
        else:
            u, s_bar = state

        synaptic_input = self.linear(input_spikes)
        # LIF integration: decay old potential + new input
        u_next = self.alpha * u + synaptic_input

        # Spike generation via surrogate gradient
        spikes = surrogate_spike(u_next, self.v_th, self.gamma)

        # Reset by subtraction (soft reset for gradient preservation)
        u_next = u_next - spikes * self.v_th

        # Update filtered spike trace
        s_bar_next = self.trace_decay * s_bar + (1.0 - self.trace_decay) * spikes

        # Local readout logits from filtered spike trace
        readout_logits = self.random_readout(s_bar_next)

        return spikes, (u_next, s_bar_next), readout_logits

    def compute_local_loss(
        self,
        readout_logits: torch.Tensor,
        targets: torch.Tensor,
        accumulated_spikes: torch.Tensor,
        time_steps: int,
    ) -> torch.Tensor:
        """
        Computes local loss: Cross-entropy on local readout + firing rate regularization.
        """
        # Cross entropy loss on local random readout predictions
        ce_loss = F.cross_entropy(readout_logits, targets)

        # Firing rate regularization: mean firing rate per neuron over the sequence
        mean_rate = accumulated_spikes.sum(dim=1).mean(dim=0) / max(time_steps, 1)  # [out_features]
        rate_loss = self.reg_rate_coef * torch.mean((mean_rate - self.target_rate) ** 2)

        total_loss = ce_loss + rate_loss
        return total_loss


class DECOLLENetwork(nn.Module):
    """
    Multi-layer SNN network equipped with DECOLLE local plasticity.
    Each layer trains independently on its own local readout loss.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        num_classes: int = 10,
        tau_m: float = 20.0,
        dt: float = 1.0,
        v_th: float = 1.0,
        gamma: float = 10.0,
        target_rate: float = 0.05,
        reg_rate_coef: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = DECOLLELayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                num_classes=num_classes,
                tau_m=tau_m,
                dt=dt,
                v_th=v_th,
                gamma=gamma,
                target_rate=target_rate,
                reg_rate_coef=reg_rate_coef,
            )
            self.layers.append(layer)

    def get_adamax_optimizers(self, lr: float = 1e-3) -> list[torch.optim.Optimizer]:
        """
        Returns an AdaMax optimizer for each layer with beta1=0.0 (memory-free local updates)
        and beta2=0.95 as required by the DECOLLE specification.
        """
        optimizers = []
        for layer in self.layers:
            opt = torch.optim.Adamax(layer.linear.parameters(), lr=lr, betas=(0.0, 0.95))
            optimizers.append(opt)
        return optimizers

    def forward_sequence(
        self, spike_sequence: torch.Tensor
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        spike_sequence: [batch_size, time_steps, in_features]
        Returns:
            layer_spikes: list of tensors [batch_size, time_steps, layer_features]
            layer_readouts: list of tensors [batch_size, num_classes] (mean readout over time)
            layer_losses: to be populated during training
        """
        batch_size, time_steps, _ = spike_sequence.shape
        device = spike_sequence.device

        current_input = spike_sequence
        all_layer_spikes = []
        all_layer_readouts = []

        for layer in self.layers:
            state = None
            step_spikes = []
            step_readouts = []

            for t in range(time_steps):
                x_t = current_input[:, t, :]
                spikes, state, readout = layer.forward_step(x_t, state)
                step_spikes.append(spikes)
                step_readouts.append(readout)

            # Stack along time dimension: [batch_size, time_steps, out_features]
            stacked_spikes = torch.stack(step_spikes, dim=1)
            # Readout averaged over sequence: [batch_size, num_classes]
            avg_readout = torch.stack(step_readouts, dim=1).mean(dim=1)

            all_layer_spikes.append(stacked_spikes)
            all_layer_readouts.append(avg_readout)

            # Detach to ensure strict locality - no gradients propagate backward across layers
            current_input = stacked_spikes.detach()

        return all_layer_spikes, all_layer_readouts
