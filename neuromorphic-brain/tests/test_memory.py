import pytest
import time
import numpy as np
from src.memory.working_memory import WorkingMemory
from src.memory.episodic_memory import EpisodicMemory, EpisodicEvent
from src.memory.semantic_memory import SemanticMemory
from src.memory.sleep_consolidation import SleepConsolidator


def test_working_memory_ttl_and_operations():
    wm = WorkingMemory(force_fallback=True)
    wm.put("test_key", {"state": "active", "value": 42}, ttl=1)
    data = wm.get("test_key")
    assert data is not None
    assert data["value"] == 42

    # Verify key deletion
    wm.put("to_delete", "val")
    assert wm.get("to_delete") == "val"
    wm.delete("to_delete")
    assert wm.get("to_delete") is None

    # Test TTL expiration
    time.sleep(1.1)
    assert wm.get("test_key") is None


def test_episodic_memory_store_and_search():
    em = EpisodicMemory(vector_dim=4, force_fallback=True)
    e1 = EpisodicEvent(
        embedding=[1.0, 0.0, 0.0, 0.0],
        context={"concept": "digit_zero", "category": "vision"},
        surprise=0.1,
    )
    e2 = EpisodicEvent(
        embedding=[0.0, 1.0, 0.0, 0.0],
        context={"concept": "digit_one", "category": "vision"},
        surprise=0.8,
    )
    em.store_event(e1)
    em.store_event(e2)

    # Search for vector close to e1
    hits = em.search_similar([0.9, 0.1, 0.0, 0.0], limit=1)
    assert len(hits) == 1
    assert hits[0][0].context["concept"] == "digit_zero"
    assert hits[0][1] > 0.8


def test_semantic_memory_knowledge_graph():
    sm = SemanticMemory()
    sm.add_concept("digit_zero", {"type": "number", "parity": "even"})
    sm.add_concept("digit_one", {"type": "number", "parity": "odd"})
    sm.add_relation("digit_zero", "digit_one", "successor", weight=1.0)

    assert sm.num_concepts() == 2
    assert sm.num_relations() == 1

    sub = sm.query_subgraph("digit_zero", depth=1)
    assert len(sub["nodes"]) == 2
    assert len(sub["edges"]) == 1
    assert sub["edges"][0]["attrs"]["relation"] == "successor"


def test_sleep_consolidation_and_identity_drift_protection():
    dim = 4
    anchor = [1.0, 0.0, 0.0, 0.0]
    em = EpisodicMemory(vector_dim=dim, force_fallback=True)
    sm = SemanticMemory()

    consolidator = SleepConsolidator(
        episodic_memory=em,
        semantic_memory=sm,
        identity_anchor_vector=anchor,
        max_drift_threshold=0.3,
    )

    # Event 1: aligned with anchor (cos_sim ~ 1.0 -> drift ~ 0.0)
    aligned_event = EpisodicEvent(
        embedding=[0.99, 0.05, 0.0, 0.0],
        context={"concept": "core_digit", "category": "math", "related_to": "foundation"},
        surprise=0.1,
    )
    # Event 2: orthogonal/divergent vector (drift ~ 1.0 > 0.3)
    drifting_event = EpisodicEvent(
        embedding=[0.0, 1.0, 0.0, 0.0],
        context={"concept": "rogue_concept", "category": "unknown"},
        surprise=0.9,
    )

    em.store_event(aligned_event)
    em.store_event(drifting_event)

    stats = consolidator.consolidate()
    assert stats["processed_events"] == 2
    assert stats["consolidated_concepts"] >= 1
    assert stats["rejected_for_drift"] >= 1
    assert sm.get_concept("core_digit") is not None
    assert sm.get_concept("rogue_concept") is None
