"""
Episodic Memory Module.
Stores time-stamped experience embeddings and associative context using Qdrant
with seamless fallback to an in-memory vector store.
"""

from typing import Any, List, Optional
import time
import uuid
import numpy as np
from pydantic import BaseModel, Field


class EpisodicEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    embedding: List[float]
    context: dict[str, Any] = Field(default_factory=dict)
    surprise: float = 0.0


class InMemoryVectorStore:
    """In-memory cosine vector store fallback when Qdrant server is unreachable."""

    def __init__(self):
        self.events: dict[str, EpisodicEvent] = {}

    def insert(self, event: EpisodicEvent) -> str:
        self.events[event.event_id] = event
        return event.event_id

    def search(self, query_vec: List[float], limit: int = 5) -> List[tuple[EpisodicEvent, float]]:
        if not self.events:
            return []
        q = np.array(query_vec, dtype=np.float32)
        norm_q = np.linalg.norm(q)
        if norm_q > 0:
            q = q / norm_q

        scored: list[tuple[EpisodicEvent, float]] = []
        for event in self.events.values():
            e = np.array(event.embedding, dtype=np.float32)
            norm_e = np.linalg.norm(e)
            if norm_e > 0:
                e = e / norm_e
            sim = float(np.dot(q, e))
            scored.append((event, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def all_events(self) -> List[EpisodicEvent]:
        return list(self.events.values())


class EpisodicMemory:
    """
    Episodic Memory interface for temporal autobiographical experience logging.
    """

    def __init__(
        self,
        collection_name: str = "brain_episodes",
        host: str = "localhost",
        port: int = 6333,
        vector_dim: int = 64,
        force_fallback: bool = False,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.is_qdrant_connected = False
        self._qdrant_client = None
        self._fallback_store = InMemoryVectorStore()

        if not force_fallback:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                client = QdrantClient(host=host, port=port, timeout=0.5, check_compatibility=False)
                # Test connection
                client.get_collections()
                # Ensure collection exists
                collections = [c.name for c in client.get_collections().collections]
                if collection_name not in collections:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
                    )
                self._qdrant_client = client
                self.is_qdrant_connected = True
            except Exception:
                self.is_qdrant_connected = False
        else:
            self.is_qdrant_connected = False

    def store_event(self, event: EpisodicEvent) -> str:
        """Stores a new episodic event record."""
        # Always update local store
        self._fallback_store.insert(event)

        if self.is_qdrant_connected and self._qdrant_client is not None:
            from qdrant_client.models import PointStruct
            try:
                self._qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=event.event_id,
                            vector=event.embedding,
                            payload={
                                "timestamp": event.timestamp,
                                "context": event.context,
                                "surprise": event.surprise,
                            },
                        )
                    ],
                )
            except Exception:
                pass
        return event.event_id

    def search_similar(self, query_vec: List[float], limit: int = 5) -> List[tuple[EpisodicEvent, float]]:
        """Returns most similar events and their cosine similarity score."""
        if self.is_qdrant_connected and self._qdrant_client is not None:
            try:
                hits = self._qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=limit,
                )
                results = []
                for hit in hits:
                    event = EpisodicEvent(
                        event_id=str(hit.id),
                        timestamp=hit.payload.get("timestamp", 0.0),
                        embedding=query_vec,  # query context
                        context=hit.payload.get("context", {}),
                        surprise=hit.payload.get("surprise", 0.0),
                    )
                    results.append((event, float(hit.score)))
                return results
            except Exception:
                pass
        return self._fallback_store.search(query_vec, limit)

    def retrieve_recent(self, limit: int = 100) -> List[EpisodicEvent]:
        events = self._fallback_store.all_events()
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
