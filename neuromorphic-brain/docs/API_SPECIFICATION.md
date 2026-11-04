# API Specification: Event Bus, Modules & Interfaces

This document describes the data contracts, Python class interfaces, and event bus schema for the Neuromorphic Brain architecture.

---

## 1. Core Data Models (`src/core/` & `src/homeostasis/`)

### StateVector
```python
class StateVector:
    curiosity: float         # [0.0, 1.0]
    error_rate: float        # [0.0, 1.0]
    resource_stress: float   # [0.0, 1.0]
    uncertainty: float       # [0.0, 1.0]

    def update(self, reward: float, error: float, load: float, uncertainty: float) -> "StateVector": ...
    def get_behavioral_mode(self) -> str: ...  # "EXPLORATION", "CAUTIOUS", "CONSERVATION", "BALANCED"
```

### EpisodicEvent
```python
class EpisodicEvent:
    event_id: str
    timestamp: float
    embedding: list[float]
    context: dict[str, Any]
    surprise: float
```

---

## 2. Component Interfaces

### DECOLLE Layer (`src/core/decolle_layer.py`)
```python
class DECOLLELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_m: float = 20.0,
        v_th: float = 1.0,
        beta_decay: float = 0.9,
        num_classes: int = 10,
        device: torch.device = None
    ): ...

    def forward_step(
        self,
        input_spikes: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Processes one time-step.
        Returns: (output_spikes, (next_potential, next_trace), local_readout_logits)
        """

    def compute_local_loss(
        self,
        local_logits: torch.Tensor,
        targets: torch.Tensor,
        spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes local loss including task cross-entropy and firing rate regularization.
        """
```

### Memory Hierarchy (`src/memory/`)

```python
class WorkingMemory:
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool: ...
    def get(self, key: str) -> Any | None: ...
    def flush(self) -> None: ...

class EpisodicMemory:
    def store_event(self, event: EpisodicEvent) -> str: ...
    def search_similar(self, query_vector: list[float], limit: int = 5) -> list[EpisodicEvent]: ...

class SemanticMemory:
    def add_concept(self, concept_id: str, attributes: dict[str, Any]) -> None: ...
    def add_relation(self, source: str, target: str, relation_type: str, weight: float = 1.0) -> None: ...
    def query_subgraph(self, concept_id: str, depth: int = 2) -> dict[str, Any]: ...

class SleepConsolidator:
    def run_consolidation_cycle(self) -> dict[str, int]:
        """
        Extracts recent episodic events, abstracts concepts into semantic graph,
        and ensures zero identity drift.
        """
```

### Predictive Coding (`src/predictive/residual_error.py`)
```python
class PredictiveCodingResidual(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (prediction, residual_error)
        """
```

### Cognitive Decision Routing (`src/routing/cdr_router.py`)
```python
class RoutingDecision(Enum):
    FAST = "FAST_SYSTEM_1"
    SLOW = "SLOW_SYSTEM_2"

class CDRRouter:
    def evaluate(
        self,
        correlation_strength: float,
        domain_crossing: float,
        stakeholder_count: int,
        uncertainty: float
    ) -> tuple[RoutingDecision, float, dict[str, float]]:
        """
        Calculates gating metric Phi(q) and routes to FAST or SLOW system.
        """
```

### World Model (`src/world_model/jepa_simulator.py`)
```python
class JEPASimulator(nn.Module):
    def predict_next(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
    def rollout(self, initial_state: torch.Tensor, action_sequence: list[torch.Tensor]) -> list[torch.Tensor]: ...
```

---

## 3. Communication Protocols
- Internal Module Communication: Python In-Process Direct Calls / Async Queues.
- Memory Layer: Redis (TCP port 6379) and Qdrant (gRPC/HTTP port 6333) with seamless fallback to In-Memory stores when external daemons are absent.
