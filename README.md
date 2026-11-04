# Digital Brain - Neuromorphic Brain Emulation

A hybrid neuromorphic cognitive framework based on local learning, spiking neural networks (DECOLLE), multi-level hierarchical memory, homeostatic regulation, predictive processing, dual-process decision routing (System 1 / System 2), and JEPA world model simulation.

---

## 🌟 Key Architecture Pillars

1. **Dynamic Synaptic Plasticity (DECOLLE Module)**:
   - Spiking Neural Network (SNN) with Leaky Integrate-and-Fire (LIF) neurons.
   - Smooth fast-sigmoid surrogate gradients.
   - Three-factor layer-wise local error using random readouts (no global Backpropagation-Through-Time).
   - AdaMax optimizer ($\beta_1 = 0.0, \beta_2 = 0.95$) and activity/activation regularizations.

2. **Multi-Level Memory Hierarchy & Sleep Consolidation**:
   - **Working Memory**: Fast volatile cache with TTL (Redis with in-memory fallback).
   - **Episodic Memory**: High-dimensional event logging with cosine similarity search (Qdrant with in-memory fallback).
   - **Semantic Memory**: Persistent structured knowledge graph (NetworkX).
   - **Sleep Consolidation**: Background NREM/REM cycle simulator abstracting episodic events into semantic concepts with anti-identity-drift safeguards.

3. **Global State Vector & Homeostatic Control**:
   - Tracks a dynamic 4D state: `[curiosity, error_rate, resource_stress, uncertainty]`.
   - Modulates high-level behavioral modes: `EXPLORATION`, `CAUTIOUS`, `CONSERVATION`, `BALANCED`.

4. **Predictive Processing (Surprise / Residual Driven)**:
   - Hierarchical autoencoding state-space module.
   - Generates top-down predictions; only residual error $E = |\text{Input} - \text{Prediction}|$ gates synaptic updates.

5. **Cognitive Decision Routing (CDR)**:
   - Dynamic dual-process gating:
     * **System 1 (Fast)**: Immediate reflex via feedforward SNN (<1ms).
     * **System 2 (Slow)**: Analytical deliberation, semantic graph reasoning, and virtual rollout simulation.
   - Evaluates correlation strength ($C_s$), domain crossing ($D_c$), stakeholder count ($S_m$), and uncertainty ($U_l$).

6. **World Model & Latent Rollout Simulation (JEPA)**:
   - Joint-Embedding Predictive Architecture predicting latent environmental transitions: $s_{t+1} = \mathcal{T}(s_t, a_t)$.
   - Evaluates multi-step virtual action rollouts without pixel-space overhead.

---

## 📁 Repository Structure

```
.
├── neuromorphic-brain/
│   ├── docs/
│   │   ├── ARCHITECTURE.md          # In-depth theoretical and mathematical foundation
│   │   ├── API_SPECIFICATION.md     # Module schemas and data contracts
│   │   └── DEPLOYMENT.md            # Docker, GPU, and service orchestration guide
│   ├── data/                        # N-MNIST and DVS dataset cache
│   ├── src/
│   │   ├── core/                    # LIF SNN, DECOLLE plasticity, dataset loader, pipeline
│   │   ├── memory/                  # Working, episodic, semantic, sleep consolidation
│   │   ├── homeostasis/             # 4D state vector & homeostatic drives
│   │   ├── predictive/              # Residual error autoencoder & synaptic gating
│   │   ├── routing/                 # CDR System 1 vs System 2 router
│   │   ├── world_model/             # JEPA latent simulator & rollout engine
│   │   └── dashboard/               # Real-time Web & Terminal interactive dashboards
│   ├── tests/                       # Complete pytest suite (19+ tests)
│   ├── Dockerfile                   # CUDA 12.1 runtime container
│   ├── docker-compose.yml           # Brain core, Redis, and Qdrant multi-service stack
│   ├── requirements.txt             # Python dependencies
│   ├── train.py                     # Training and simulation entry point
│   └── dashboard_app.py             # Live Monitoring Dashboard (Web UI & Terminal Monitor)
└── README.md
```

---

## 🚀 Quick Start

### 1. Native Python Environment

```bash
# Clone and enter the project folder
cd neuromorphic-brain

# Install dependencies (PyTorch, Qdrant, Redis, NetworkX, FastAPI, etc.)
pip install -r requirements.txt

# Run the complete cognitive training & sleep consolidation cycle
python train.py

# Launch the live Real-Time Web & Terminal Dashboard
python dashboard_app.py --port 8000
```
Open your browser at `http://localhost:8000` to view real-time state vectors, live decision routes, memory logs, and interactive controls!

### 2. Docker & Docker Compose

```bash
cd neuromorphic-brain
docker compose up --build
```

---

## 🧪 Running Tests

Execute the automated test suite covering all modules:
```bash
pytest neuromorphic-brain/tests -v
```

---

## 📊 Live Monitoring Dashboard

The monitoring system provides:
- **Real-Time Visual Gauges**: Curiosity, Error Rate, Resource Stress, Uncertainty.
- **Route Tracking**: Instant display of System 1 (Fast reflex) vs System 2 (Slow deliberation) decisions.
- **Memory Inspector**: Live view of working memory keys, episodic events, and semantic graph concept count.
- **REST & SSE Endpoints**: Easily hook external tools or custom dashboards into the brain state stream.
