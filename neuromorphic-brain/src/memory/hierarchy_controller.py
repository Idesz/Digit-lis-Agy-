"""
Memory Hierarchy Controller and Sleep Consolidation Worker.
Matches Pillar 2 from software_and_code_specification.md.
"""

from typing import Any, Dict, List, Optional
import time


class MemoryHierarchyController:
    """
    Three-level memory controller orchestrating Working Memory, Episodic Memory, and Semantic Graph.
    """

    def __init__(self) -> None:
        self.working_memory: Dict[str, Any] = {}
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_graph: Dict[str, List[str]] = {}

    def write_working_memory(self, key: str, data: Any) -> None:
        self.working_memory[key] = data

    def store_episodic_event(self, event_type: str, payload: Dict[str, Any], vector_embedding: List[float]) -> None:
        event_record = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload,
            "embedding": vector_embedding,
            "consolidated": False,
        }
        self.episodic_memory.append(event_record)

    def update_semantic_graph(self, concept: str, related_concept: str) -> None:
        if concept not in self.semantic_graph:
            self.semantic_graph[concept] = []
        if related_concept not in self.semantic_graph[concept]:
            self.semantic_graph[concept].append(related_concept)


class SleepConsolidationWorker:
    """
    Cyclic sleep consolidation worker abstracting episodic experiences into the semantic graph.
    """

    def __init__(self, memory_controller: MemoryHierarchyController, cycle_threshold: int = 100) -> None:
        self.memory = memory_controller
        self.cycle_threshold = cycle_threshold

    def should_trigger_consolidation(self, current_cycle_count: int, is_idle: bool) -> bool:
        if is_idle or (current_cycle_count > 0 and current_cycle_count % self.cycle_threshold == 0):
            return True
        return False

    def run_sleep_consolidation(self, noise_threshold_age: float = 86400.0) -> None:
        now = time.time()
        unconsolidated_events = [e for e in self.memory.episodic_memory if not e["consolidated"]]

        for event in unconsolidated_events:
            payload = event.get("payload", {})
            subject = payload.get("subject")
            predicate = payload.get("predicate")

            if subject and predicate:
                self.memory.update_semantic_graph(subject, predicate)

            event["consolidated"] = True

        self.memory.episodic_memory = [
            e for e in self.memory.episodic_memory
            if not (not e["consolidated"] and (now - e["timestamp"]) > noise_threshold_age)
        ]

        self.memory.working_memory.clear()
