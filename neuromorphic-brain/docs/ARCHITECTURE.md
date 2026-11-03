# Architecture Specification: Neuromorphic Brain Emulation

This document details the theoretical, mathematical, and architectural foundations of the hybrid neuromorphic cognitive framework based on 6 core pillars.

---

## 1. Dynamic Synaptic Plasticity (DECOLLE Module)

### Theoretical Foundation
Standard Spiking Neural Networks (SNNs) trained with Backpropagation-Through-Time (BPTT) suffer from severe memory bottlenecks and biological implausibility due to spatial and temporal non-locality. The **DECOLLE** (Deep Continuous Local Learning) paradigm replaces global backpropagation with layer-wise, three-factor local plasticity.

### Neuron Dynamics: Leaky Integrate-and-Fire (LIF)
Each spiking layer models LIF dynamics discretized at time-step $\Delta t$:
$$U_i[t] = \alpha U_i[t-1] + \sum_j W_{ij} S_j[t-1] - S_i[t-1] V_{th}$$
Where:
- $U_i[t]$ is the membrane potential of neuron $i$.
- $\alpha = \exp(-\Delta t / \tau_m)$ is the membrane decay constant ($0 < \alpha < 1$).
- $W_{ij}$ is the synaptic connection weight from presynaptic neuron $j$.
- $S_j[t] \in \{0, 1\}$ is the spike event: $S_i[t] = \Theta(U_i[t] - V_{th})$, where $\Theta$ is the Heaviside step function.
- $V_{th}$ is the firing threshold.

### Surrogate Gradient
Because $\Theta'(x)$ is non-differentiable (Dirac delta function), a smooth surrogate gradient is used during the backward pass:
$$\sigma'(U) = \frac{1}{(1 + \gamma |U - V_{th}|)^2}$$
where $\gamma$ is a smoothing scale factor.

### Local Learning Rule & Random Readout
Each intermediate layer $l$ is equipped with a fixed, random linear readout matrix $P^l$. At time step $t$, the layer produces a local error:
$$e^l[t] = \hat{y}^l[t] - y[t] \quad \text{where} \quad \hat{y}^l[t] = P^l \bar{S}^l[t]$$
Here $\bar{S}^l[t]$ is the exponentially smoothed trace of spikes:
$$\bar{S}^l[t] = \beta \bar{S}^l[t-1] + (1 - \beta) S^l[t]$$
The weight update $\Delta W^l$ depends solely on local pre-synaptic trace and local post-synaptic error, eliminating backward gradient transport across layers:
$$\Delta W^l[t] \propto - e^l[t] (P^l)^T \odot \sigma'(U^l[t]) \otimes \bar{S}^{l-1}[t]$$

### Optimizer & Regularization
- **Optimizer**: AdaMax configured with $\beta_1 = 0.0$ (memory-free momentum for streaming updates) and $\beta_2 = 0.95$.
- **Activity & Activation Regularization**: Firing rate penalty $L_{reg} = \lambda_{rate} (\frac{1}{T}\sum_t S_i[t] - f_{target})^2$ prevents both quiescent networks and epileptic over-firing.

---

## 2. Multi-Level Memory Hierarchy & Sleep Consolidation

### Memory Continuum
1. **Working Memory (Redis Cache / In-Memory)**:
   - High-throughput, sub-millisecond read/write access.
   - Holds recent sensory observations, active attention window, and immediate context with a strict Time-To-Live (TTL).
2. **Episodic Memory (Vector Store - Qdrant / In-Memory Vector Store)**:
   - Stores temporally indexed event embeddings $\mathbf{e}_t \in \mathbb{R}^D$ alongside metadata (timestamp, context, surprise metrics).
   - Enables Nearest Neighbor associative recall using Cosine / Dot Product similarity.
3. **Semantic Memory (NetworkX / Graph Database)**:
   - Structured factual knowledge graph: Nodes represent entities/concepts; edges represent directed semantic relationships and transition likelihoods.

### Sleep Consolidation Process
Biological brains consolidate fragile episodic experiences into stable cortical semantic representations during sleep without catastrophic forgetting.
- The **Consolidation Worker** runs asynchronously in background cycles.
- It clusters episodic memory traces and abstracts frequent relational structures into semantic graph nodes and edges.
- Regularization constraints prevent **identity drift** by evaluating cosine drift against anchor embeddings before permanently modifying the semantic core.

---

## 3. Global State Vector & Homeostatic Control

The agent regulates its intrinsic motivations via a continuous 4D homeostatic state vector:
$$\mathbf{H} = [C, E, R, U]^T$$
Where:
- $C \in [0, 1]$: **Curiosity level** (driven by novel sensory embeddings and prediction entropy).
- $E \in [0, 1]$: **Error rate / Performance stress** (running average of task or predictive error).
- $R \in [0, 1]$: **Resource stress** (computational load, memory saturation, latency bounds).
- $U \in [0, 1]$: **Uncertainty / Epistemic doubt** (variance across internal predictor heads).

### Behavioral Regimes
- **Exploration Mode** ($C > \theta_C$ and $E < \theta_E$): Expands search policies and increases sampling of novel input spaces.
- **Cautious / Deliberation Mode** ($E > \theta_E$ or $U > \theta_U$): Forces routing to deliberate System 2 evaluation, lowering execution rate.
- **Conservation Mode** ($R > \theta_R$): Prunes working memory traces and skips deep simulations to conserve computational resources.

---

## 4. Predictive Processing (Residual Error Driving)

Hierarchical predictive processing posits that higher layers do not transmit raw sensory data; instead, top-down generative models predict lower-layer representations, and only unexplained residuals (surprise) are propagated upward.

- Prediction: $\hat{\mathbf{x}}_t = f_{pred}(\mathbf{z}_t)$
- Residual Error: $\mathbf{r}_t = |\mathbf{x}_t - \hat{\mathbf{x}}_t|$
- Synaptic plastic updates are gated by residual magnitude:
  $$\Delta W \propto g(\|\mathbf{r}_t\|_2) \cdot \nabla W$$
  where $g(r) \approx 0$ when prediction error is negligible, saving energy and preserving learned weights.

---

## 5. Cognitive Decision Routing (CDR - System 1 vs. System 2)

Dual-process cognitive architecture balances rapid habitual intuition with deep analytical deliberation.

### Evaluation Dimensions
The router assesses queries/tasks along 4 dimensions:
1. $C_s$: Correlation strength with past known experiences (from Episodic Memory).
2. $D_c$: Domain crossing / Cross-domain abstract reasoning requirements.
3. $S_m$: Number of interdependent stakeholders / entities affected.
4. $U_l$: Uncertainty level derived from predictive dispersion.

### Gating Function
A composite complexity score is computed:
$$\Phi(q) = w_1 (1 - C_s) + w_2 D_c + w_3 S_m + w_4 U_l$$
- If $\Phi(q) < \tau$: **System 1 (Fast Route)** is triggered (SNN feedforward reflex, sub-millisecond latency).
- If $\Phi(q) \ge \tau$: **System 2 (Slow Route)** is triggered (semantic graph traversal, JEPA latent rollout simulation, multi-step deliberation).

---

## 6. World Model & Latent Rollout Simulation (JEPA)

### Joint-Embedding Predictive Architecture (JEPA)
Instead of predicting high-dimensional pixel/sensor spaces (which wastes capacity on irrelevant noise), the JEPA world model operates entirely within abstract latent embeddings:
$$\mathbf{s}_{t+1}^{latent} = \mathcal{T}(\mathbf{s}_t^{latent}, \mathbf{a}_t)$$
Where:
- $\mathbf{s}_t^{latent} \in \mathbb{R}^d$ is the state representation extracted by the sensory encoder.
- $\mathbf{a}_t$ is candidate action or external environmental transition.
- $\mathcal{T}$ is the latent transition predictor.

### Rollout Simulation
System 2 leverages the world model to simulate $K$-step virtual rollouts:
$$\{\mathbf{s}_{t+1}^{latent}, \dots, \mathbf{s}_{t+K}^{latent}\}$$
Rollout outcomes are scored using the homeostatic drive function to select optimal trajectories before physical action execution.
