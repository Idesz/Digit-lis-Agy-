"""
Semantic Memory Module.
Structured knowledge graph using NetworkX for persistent concept representation,
relational inference, and topological queries.
"""

from typing import Any, Dict, List, Optional
import networkx as nx


class SemanticMemory:
    """
    Graph-based semantic memory representing enduring factual relations.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_concept(self, concept_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Adds or updates a semantic entity/concept node."""
        attrs = attributes or {}
        if self.graph.has_node(concept_id):
            self.graph.nodes[concept_id].update(attrs)
        else:
            self.graph.add_node(concept_id, **attrs)

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Creates a directed semantic relationship edge."""
        if not self.graph.has_node(source):
            self.add_concept(source)
        if not self.graph.has_node(target):
            self.add_concept(target)

        edge_attrs = attributes or {}
        edge_attrs.update({"relation": relation_type, "weight": weight})
        self.graph.add_edge(source, target, **edge_attrs)

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(concept_id):
            return dict(self.graph.nodes[concept_id])
        return None

    def query_subgraph(self, concept_id: str, depth: int = 1) -> Dict[str, Any]:
        """Extracts ego subgraph around a concept for associative reasoning."""
        if not self.graph.has_node(concept_id):
            return {"nodes": [], "edges": []}

        subgraph_nodes = nx.single_source_shortest_path_length(self.graph, concept_id, cutoff=depth).keys()
        sub = self.graph.subgraph(subgraph_nodes)

        nodes = [{"id": n, "attrs": sub.nodes[n]} for n in sub.nodes()]
        edges = [{"source": u, "target": v, "attrs": sub.edges[u, v]} for u, v in sub.edges()]
        return {"nodes": nodes, "edges": edges}

    def num_concepts(self) -> int:
        return self.graph.number_of_nodes()

    def num_relations(self) -> int:
        return self.graph.number_of_edges()
