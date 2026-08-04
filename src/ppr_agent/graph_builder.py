from __future__ import annotations
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import igraph as ig
from tqdm import tqdm
from .openie_extractor import load_openie_results
from .schema import (
    EdgeType,
    GraphBuildResult,
    GraphEdge,
    GraphNode,
    OpenIEDoc,
    Triple,
    compute_mdhash_id,
    normalize_text,
    triple_to_text,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphBuilderConfig:
    output_dir: str

    graph_filename: str = "graph.pickle"
    metadata_filename: str = "graph_metadata.json"

    directed: bool = False

    # Edge weights
    relation_edge_weight: float = 1.0
    passage_entity_edge_weight: float = 1.0
    include_ner_entities: bool = False
    add_reverse_relation_edges: bool = True
    add_reverse_passage_edges: bool = True
    force_rebuild: bool = True


def entity_id_from_text(entity_text: str) -> str:
    normalized = normalize_text(entity_text)
    return compute_mdhash_id(normalized, prefix="entity-")


def passage_id_from_text(passage_text: str) -> str:
    return compute_mdhash_id(str(passage_text), prefix="chunk-")


def triple_id_from_parts(subject: str, predicate: str, object_: str) -> str:
    s = normalize_text(subject)
    p = normalize_text(predicate)
    o = normalize_text(object_)
    return compute_mdhash_id(str((s, p, o)), prefix="fact-")


def canonical_triple(triple: Triple) -> Tuple[str, str, str]:
    return (
        normalize_text(triple.subject),
        normalize_text(triple.predicate),
        normalize_text(triple.object),
    )


def _safe_sorted(values: Iterable[str]) -> List[str]:
    return sorted(set(str(v) for v in values))


class PPRAgentGraphBuilder:
    def __init__(self, config: GraphBuilderConfig):
        self.config = config

        os.makedirs(self.config.output_dir, exist_ok=True)

        self.graph_path = os.path.join(
            self.config.output_dir,
            self.config.graph_filename,
        )
        self.metadata_path = os.path.join(
            self.config.output_dir,
            self.config.metadata_filename,
        )

        self.reset_state()


    def reset_state(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edge_stats: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        self.entity_id_to_text: Dict[str, str] = {}
        self.entity_text_to_id: Dict[str, str] = {}

        self.passage_id_to_text: Dict[str, str] = {}
        self.passage_id_to_metadata: Dict[str, Dict[str, Any]] = {}

        self.entity_to_passage_ids: Dict[str, Set[str]] = defaultdict(set)
        self.passage_to_entity_ids: Dict[str, Set[str]] = defaultdict(set)

        self.triple_records: Dict[str, Dict[str, Any]] = {}


    def build_from_openie_file(self, openie_path: str) -> GraphBuildResult:
        docs = load_openie_results(openie_path)
        return self.build_from_openie_docs(docs)

    def build_from_openie_docs(self, docs: Sequence[OpenIEDoc]) -> GraphBuildResult:
        self.reset_state()

        logger.info("Building PPR-agent graph from %d OpenIE docs.", len(docs))

        for doc in tqdm(docs, desc="Building graph"):
            self._add_openie_doc(doc)

        graph = self._to_igraph()
        graph.write_pickle(self.graph_path)

        metadata = self._build_metadata()
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        result = GraphBuildResult(
            graph_path=self.graph_path,
            num_entity_nodes=sum(1 for node in self.nodes.values() if node.node_type == "entity"),
            num_passage_nodes=sum(1 for node in self.nodes.values() if node.node_type == "passage"),
            num_relation_edges=sum(
                1 for edge in self.edge_stats.values() if edge["edge_type"] == "relation"
            ),
            num_passage_edges=sum(
                1 for edge in self.edge_stats.values() if edge["edge_type"] in {"contains", "mentioned_in"}
            ),
            metadata={
                "metadata_path": self.metadata_path,
                "num_total_nodes": len(self.nodes),
                "num_total_edges": len(self.edge_stats),
                "num_unique_triples": len(self.triple_records),
            },
        )

        logger.info(
            "Graph saved: nodes=%d, edges=%d, triples=%d",
            len(self.nodes),
            len(self.edge_stats),
            len(self.triple_records),
        )

        return result


    def _add_openie_doc(self, doc: OpenIEDoc) -> None:
        passage_id = doc.passage_id or passage_id_from_text(doc.passage)
        passage_text = doc.passage

        self._add_passage_node(
            passage_id=passage_id,
            passage_text=passage_text,
            metadata=doc.metadata,
        )

        triple_entities: Set[str] = set()

        for triple in doc.extracted_triples:
            self._add_triple(
                triple=triple,
                source_passage_id=passage_id,
            )

            s, _, o = canonical_triple(triple)
            triple_entities.add(s)
            triple_entities.add(o)

        if self.config.include_ner_entities:
            for ent in doc.extracted_entities:
                normalized_ent = normalize_text(ent)
                if normalized_ent:
                    triple_entities.add(normalized_ent)

        for entity_text in triple_entities:
            entity_id = self._add_entity_node(entity_text)
            self._add_passage_entity_edge(
                passage_id=passage_id,
                entity_id=entity_id,
            )

    def _add_passage_node(
        self,
        passage_id: str,
        passage_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if passage_id not in self.nodes:
            self.nodes[passage_id] = GraphNode(
                node_id=passage_id,
                node_type="passage",
                text=passage_text,
                metadata=metadata or {},
            )

        self.passage_id_to_text[passage_id] = passage_text
        self.passage_id_to_metadata[passage_id] = metadata or {}

    def _add_entity_node(self, entity_text: str) -> str:
        normalized = normalize_text(entity_text)
        entity_id = entity_id_from_text(normalized)

        if entity_id not in self.nodes:
            self.nodes[entity_id] = GraphNode(
                node_id=entity_id,
                node_type="entity",
                text=normalized,
                metadata={},
            )

        self.entity_id_to_text[entity_id] = normalized
        self.entity_text_to_id[normalized] = entity_id

        return entity_id

    def _add_triple(self, triple: Triple, source_passage_id: str) -> None:
        subject, predicate, object_ = canonical_triple(triple)

        if not subject or not predicate or not object_:
            return

        subject_id = self._add_entity_node(subject)
        object_id = self._add_entity_node(object_)

        triple_id = triple_id_from_parts(subject, predicate, object_)

        if triple_id not in self.triple_records:
            self.triple_records[triple_id] = {
                "triple_id": triple_id,
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "text": triple_to_text(subject, predicate, object_),
                "source_passage_ids": [],
                "metadata": dict(triple.metadata),
            }

        if source_passage_id not in self.triple_records[triple_id]["source_passage_ids"]:
            self.triple_records[triple_id]["source_passage_ids"].append(source_passage_id)

        self._add_relation_edge(
            subject_id=subject_id,
            object_id=object_id,
            relation=predicate,
            source_passage_id=source_passage_id,
        )

        self.entity_to_passage_ids[subject_id].add(source_passage_id)
        self.entity_to_passage_ids[object_id].add(source_passage_id)

    def _add_relation_edge(
        self,
        subject_id: str,
        object_id: str,
        relation: str,
        source_passage_id: str,
    ) -> None:
        self._add_edge(
            src=subject_id,
            dst=object_id,
            edge_type="relation",
            weight=self.config.relation_edge_weight,
            relation=relation,
            source_passage_id=source_passage_id,
        )

        if self.config.directed and self.config.add_reverse_relation_edges:
            self._add_edge(
                src=object_id,
                dst=subject_id,
                edge_type="relation",
                weight=self.config.relation_edge_weight,
                relation=f"reverse:{relation}",
                source_passage_id=source_passage_id,
            )

    def _add_passage_entity_edge(self, passage_id: str, entity_id: str) -> None:
        self.passage_to_entity_ids[passage_id].add(entity_id)
        self.entity_to_passage_ids[entity_id].add(passage_id)

        self._add_edge(
            src=passage_id,
            dst=entity_id,
            edge_type="contains",
            weight=self.config.passage_entity_edge_weight,
            relation="contains",
            source_passage_id=passage_id,
        )

        if self.config.directed and self.config.add_reverse_passage_edges:
            self._add_edge(
                src=entity_id,
                dst=passage_id,
                edge_type="mentioned_in",
                weight=self.config.passage_entity_edge_weight,
                relation="mentioned_in",
                source_passage_id=passage_id,
            )

    def _edge_key(self, src: str, dst: str, edge_type: str) -> Tuple[str, str, str]:
        if self.config.directed:
            return src, dst, edge_type

        a, b = sorted([src, dst])
        return a, b, edge_type

    def _add_edge(
        self,
        src: str,
        dst: str,
        edge_type: EdgeType,
        weight: float,
        relation: Optional[str] = None,
        source_passage_id: Optional[str] = None,
    ) -> None:
        if src == dst:
            return

        key = self._edge_key(src, dst, edge_type)

        if key not in self.edge_stats:
            if self.config.directed:
                final_src, final_dst = src, dst
            else:
                final_src, final_dst = key[0], key[1]

            self.edge_stats[key] = {
                "src": final_src,
                "dst": final_dst,
                "edge_type": edge_type,
                "weight": 0.0,
                "relations": set(),
                "source_passage_ids": set(),
            }

        self.edge_stats[key]["weight"] += float(weight)

        if relation:
            self.edge_stats[key]["relations"].add(relation)

        if source_passage_id:
            self.edge_stats[key]["source_passage_ids"].add(source_passage_id)


    def _to_igraph(self) -> ig.Graph:
        graph = ig.Graph(directed=self.config.directed)

        node_ids = list(self.nodes.keys())
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        vertex_attrs: Dict[str, List[Any]] = {
            "name": [],
            "node_type": [],
            "content": [],
            "text": [],
            "metadata": [],
        }

        for node_id in node_ids:
            node = self.nodes[node_id]
            vertex_attrs["name"].append(node.node_id)
            vertex_attrs["node_type"].append(node.node_type)
            vertex_attrs["content"].append(node.text)
            vertex_attrs["text"].append(node.text)
            vertex_attrs["metadata"].append(dict(node.metadata))

        graph.add_vertices(len(node_ids), attributes=vertex_attrs)

        edges: List[Tuple[int, int]] = []
        edge_attrs: Dict[str, List[Any]] = {
            "weight": [],
            "edge_type": [],
            "relation": [],
            "relations": [],
            "source_passage": [],
            "source_passage_ids": [],
        }

        for edge in self.edge_stats.values():
            src = edge["src"]
            dst = edge["dst"]

            if src not in node_id_to_idx or dst not in node_id_to_idx:
                continue

            edges.append((node_id_to_idx[src], node_id_to_idx[dst]))

            relations = _safe_sorted(edge["relations"])
            source_passage_ids = _safe_sorted(edge["source_passage_ids"])

            edge_attrs["weight"].append(float(edge["weight"]))
            edge_attrs["edge_type"].append(edge["edge_type"])
            edge_attrs["relations"].append(relations)
            edge_attrs["relation"].append(relations[0] if relations else None)
            edge_attrs["source_passage_ids"].append(source_passage_ids)
            edge_attrs["source_passage"].append(
                source_passage_ids[0] if source_passage_ids else None
            )

        if edges:
            graph.add_edges(edges, attributes=edge_attrs)

        return graph


    def _build_metadata(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "entity_id_to_text": dict(self.entity_id_to_text),
            "entity_text_to_id": dict(self.entity_text_to_id),
            "passage_id_to_text": dict(self.passage_id_to_text),
            "passage_id_to_metadata": dict(self.passage_id_to_metadata),
            "entity_to_passage_ids": {
                entity_id: _safe_sorted(passage_ids)
                for entity_id, passage_ids in self.entity_to_passage_ids.items()
            },
            "passage_to_entity_ids": {
                passage_id: _safe_sorted(entity_ids)
                for passage_id, entity_ids in self.passage_to_entity_ids.items()
            },
            "triple_records": list(self.triple_records.values()),
            "num_entity_nodes": sum(
                1 for node in self.nodes.values() if node.node_type == "entity"
            ),
            "num_passage_nodes": sum(
                1 for node in self.nodes.values() if node.node_type == "passage"
            ),
            "num_total_nodes": len(self.nodes),
            "num_total_edges": len(self.edge_stats),
            "num_unique_triples": len(self.triple_records),
        }


@dataclass
class GraphArtifact:
    graph: ig.Graph
    metadata: Dict[str, Any]
    graph_path: str
    metadata_path: str


def load_graph_artifact(output_dir: str) -> GraphArtifact:
    graph_path = os.path.join(output_dir, "graph.pickle")
    metadata_path = os.path.join(output_dir, "graph_metadata.json")

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Graph metadata file not found: {metadata_path}")

    graph = ig.Graph.Read_Pickle(graph_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return GraphArtifact(
        graph=graph,
        metadata=metadata,
        graph_path=graph_path,
        metadata_path=metadata_path,
    )


def build_graph_from_openie_file(
    openie_path: str,
    output_dir: str,
    include_ner_entities: bool = False,
    directed: bool = False,
) -> GraphBuildResult:
    config = GraphBuilderConfig(
        output_dir=output_dir,
        include_ner_entities=include_ner_entities,
        directed=directed,
    )
    builder = PPRAgentGraphBuilder(config)
    return builder.build_from_openie_file(openie_path)
