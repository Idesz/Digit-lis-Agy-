"""
Synaptic Graph Manager (Hebbian Local Learning & Continuous Pruning).
Corresponds directly to Pillar 1 from software_and_code_specification.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class Node:
    """Represents a node (neuron or abstract concept) in the dynamic graph."""
    node_id: str
    activation_history: List[float] = field(default_factory=list)


@dataclass
class Edge:
    """Represents a dynamic synaptic connection between two nodes."""
    source_id: str
    target_id: str
    weight: float
    last_activated: float


class SynapticGraphManager:
    """
    Distributed directed synaptic graph manager executing Hebbian local learning and continuous pruning.
    """

    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.01, initial_weight: float = 0.2) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}  # key format: "source_id->target_id"
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.initial_weight = initial_weight

    def add_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id=node_id)

    def trigger_activation(self, node_id: str, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        if node_id in self.nodes:
            self.nodes[node_id].activation_history.append(timestamp)

    def update_local_learning(self, node_a: str, node_b: str, time_window: float = 0.5) -> None:
        if node_a not in self.nodes or node_b not in self.nodes:
            return

        now = time.time()
        history_a = [t for t in self.nodes[node_a].activation_history if now - t <= time_window]
        history_b = [t for t in self.nodes[node_b].activation_history if now - t <= time_window]

        if history_a and history_b:
            edge_key = f"{node_a}->{node_b}"
            if edge_key not in self.edges:
                self.edges[edge_key] = Edge(
                    source_id=node_a,
                    target_id=node_b,
                    weight=self.initial_weight,
                    last_activated=now,
                )
            else:
                current_weight = self.edges[edge_key].weight
                self.edges[edge_key].weight = current_weight + self.learning_rate * (1.0 - current_weight)
                self.edges[edge_key].last_activated = now

    def prune_inactive_edges(self, threshold: float = 0.1, max_idle_time: float = 10.0) -> None:
        now = time.time()
        keys_to_delete: List[str] = []

        for edge_key, edge in self.edges.items():
            idle_duration = now - edge.last_activated
            if idle_duration > max_idle_time:
                edge.weight -= self.decay_factor * (idle_duration / max_idle_time)

            if edge.weight < threshold:
                keys_to_delete.append(edge_key)

        for key in keys_to_delete:
            del self.edges[key]
