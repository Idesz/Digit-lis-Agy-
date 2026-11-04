"""
Sleep Consolidation Worker.
Abstracts episodic memory into the semantic knowledge graph during simulated sleep cycles.
Guards against identity drift using cosine similarity verification against core identity anchors.
"""

from typing import Dict, List, Optional
import numpy as np
from src.memory.episodic_memory import EpisodicMemory, EpisodicEvent
from src.memory.semantic_memory import SemanticMemory


class SleepConsolidator:
    """
    Background consolidation process simulating NREM/REM sleep cycles.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        identity_anchor_vector: Optional[List[float]] = None,
        max_drift_threshold: float = 0.35,
    ):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.max_drift_threshold = max_drift_threshold

        # Core anchor vector defining agent baseline identity / knowledge centroid
        # If None, dynamic anchoring: initial anchor learned or zeroed until established
        if identity_anchor_vector is not None:
            vec = np.array(identity_anchor_vector, dtype=np.float32)
            norm = np.linalg.norm(vec)
            self.identity_anchor = (vec / norm) if norm > 0 else None
        else:
            self.identity_anchor = None

    def calculate_drift(self, candidate_vector: np.ndarray) -> float:
        """Computes cosine distance (1 - cosine_similarity) from identity anchor."""
        norm = np.linalg.norm(candidate_vector)
        if norm == 0:
            return 0.0
        unit_cand = candidate_vector / norm

        if self.identity_anchor is None:
            # Anchor initializes with the first stable observation
            self.identity_anchor = unit_cand.copy()
            return 0.0

        cos_sim = float(np.dot(self.identity_anchor, unit_cand))
        return max(0.0, 1.0 - cos_sim)

    def consolidate(self, batch_size: int = 50) -> Dict[str, int]:
        """
        Executes a consolidation pass:
        1. Retrieves un-consolidated recent episodic events.
        2. Identifies high-surprise or recurring concepts.
        3. Integrates them into the semantic graph while verifying no identity drift occurs.
        """
        recent_events = self.episodic.retrieve_recent(limit=batch_size)
        consolidated_concepts = 0
        consolidated_relations = 0
        rejected_for_drift = 0

        for event in recent_events:
            vec = np.array(event.embedding, dtype=np.float32)
            drift = self.calculate_drift(vec)

            # Safeguard: prevent identity drift
            if drift > self.max_drift_threshold:
                rejected_for_drift += 1
                continue

            concept_name = event.context.get("concept")
            category = event.context.get("category", "general")

            if concept_name:
                self.semantic.add_concept(
                    concept_id=concept_name,
                    attributes={
                        "category": category,
                        "surprise": event.surprise,
                        "last_consolidated": event.timestamp,
                    },
                )
                consolidated_concepts += 1

                related_to = event.context.get("related_to")
                if related_to:
                    self.semantic.add_relation(
                        source=concept_name,
                        target=related_to,
                        relation_type="associated_with",
                        weight=float(1.0 / (1.0 + event.surprise)),
                    )
                    consolidated_relations += 1

        return {
            "processed_events": len(recent_events),
            "consolidated_concepts": consolidated_concepts,
            "consolidated_relations": consolidated_relations,
            "rejected_for_drift": rejected_for_drift,
        }
