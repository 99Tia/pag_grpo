from __future__ import annotations
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import igraph as ig
import numpy as np
from .graph_builder import entity_id_from_text, load_graph_artifact
from .schema import (
    CandidateTriple,
    FilteredTriple,
    PPRSeedInfo,
    RetrievedPassage,
    SearchGraphRequest,
    SearchGraphResult,
    Triple,
    normalize_text,
)
from .triple_filter import TripleFilter, TripleFilterConfig
from .triple_index import TripleIndex, TripleIndexConfig, min_max_normalize

logger = logging.getLogger(__name__)


@dataclass
class PPRSearchConfig:
    graph_dir: str
    index_dir: str
    graph_metadata_path: Optional[str] = None

    # Triple retrieval
    top_k_candidate_triples: int = 50

    # LLM/DSPy triple filtering
    enable_triple_filter: bool = True
    fallback_to_candidates_if_filter_empty: bool = True
    max_filtered_triples: Optional[int] = None

    # PPR reset construction
    linking_top_k: Optional[int] = 50
    passage_node_weight: float = 0.05
    seed_entity_weight: float = 1.0
    filtered_triple_weight: float = 1.0

    # NEW:
    # Directly seed source passages of LLM-selected triples.
    # This makes LLM triple filtering affect PPR more strongly.
    selected_triple_passage_weight: float = 1.0
    max_selected_triple_passage_seeds: int = 20

    # Dense passage reset
    dense_reset_top_k: int = 50

    # PageRank
    damping: float = 0.5

    # Output
    top_k_passages: int = 5

    # Debug
    save_debug: bool = False
    debug_dir: Optional[str] = None


@dataclass
class PPRSearchDebugInfo:
    retrieval_query: str
    filter_query: str
    num_candidate_triples: int
    num_filtered_triples: int
    num_entity_seeds: int
    num_passage_seeds: int
    num_dense_passage_seeds: int
    num_selected_triple_passage_seeds: int
    reset_mass: float
    damping: float
    passage_node_weight: float
    selected_triple_passage_weight: float

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

def _top_k_dict(input_dict: Dict[str, float], k: Optional[int]) -> Dict[str, float]:
    if k is None or k <= 0:
        return dict(input_dict)

    return dict(
        sorted(input_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    )

def _as_filtered_from_candidate(candidate: CandidateTriple, rank: int) -> FilteredTriple:
    return FilteredTriple(
        triple=candidate.triple,
        original_index=candidate.index,
        original_score=candidate.score,
        filter_rank=rank,
        filter_confidence=None,
    )


def _triple_entities(triple: Triple) -> Tuple[str, str]:
    return normalize_text(triple.subject), normalize_text(triple.object)


def _dedupe_strings(items: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()

    for item in items:
        text = str(item).strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        output.append(text)
        seen.add(key)

    return output


def _truncate(text: str, max_chars: int = 500) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


class PPRSearchEngine:

    def __init__(
        self,
        config: PPRSearchConfig,
        embedding_model: Any = None,
        triple_filter_config: Optional[TripleFilterConfig] = None,
    ):
        self.config = config
        self.embedding_model = embedding_model

        if self.config.graph_metadata_path is None:
            self.config.graph_metadata_path = os.path.join(
                self.config.graph_dir,
                "graph_metadata.json",
            )

        self.graph_artifact = load_graph_artifact(self.config.graph_dir)
        self.graph: ig.Graph = self.graph_artifact.graph
        self.graph_metadata: Dict[str, Any] = self.graph_artifact.metadata

        self.triple_index = TripleIndex(
            TripleIndexConfig(
                index_dir=self.config.index_dir,
                graph_metadata_path=self.config.graph_metadata_path,
            ),
            embedding_model=embedding_model,
        )
        self.triple_index.load()

        self.node_name_to_vertex_idx: Dict[str, int] = {
            vertex["name"]: idx
            for idx, vertex in enumerate(self.graph.vs)
        }

        self.passage_ids: List[str] = list(
            self.graph_metadata.get("passage_id_to_text", {}).keys()
        )
        self.passage_node_idxs: List[int] = [
            self.node_name_to_vertex_idx[pid]
            for pid in self.passage_ids
            if pid in self.node_name_to_vertex_idx
        ]

        self.entity_to_passage_ids: Dict[str, List[str]] = {
            entity_id: list(passage_ids)
            for entity_id, passage_ids in self.graph_metadata.get(
                "entity_to_passage_ids",
                {},
            ).items()
        }

        self.entity_text_to_id: Dict[str, str] = {
            normalize_text(k): v
            for k, v in self.graph_metadata.get("entity_text_to_id", {}).items()
        }

        self.passage_id_to_text: Dict[str, str] = dict(
            self.graph_metadata.get("passage_id_to_text", {})
        )
        self.passage_id_to_metadata: Dict[str, Dict[str, Any]] = dict(
            self.graph_metadata.get("passage_id_to_metadata", {})
        )

        self.triple_filter: Optional[TripleFilter] = None
        if self.config.enable_triple_filter:
            if triple_filter_config is None:
                triple_filter_config = TripleFilterConfig(
                    enabled=True,
                    fallback_to_input_if_empty=self.config.fallback_to_candidates_if_filter_empty,
                    max_output_triples=self.config.max_filtered_triples,
                )
            else:
                triple_filter_config.fallback_to_input_if_empty = (
                    self.config.fallback_to_candidates_if_filter_empty
                )
                triple_filter_config.max_output_triples = (
                    self.config.max_filtered_triples
                )

            self.triple_filter = TripleFilter(triple_filter_config)

        logger.info(
            "Initialized PPRSearchEngine: nodes=%d, edges=%d, passages=%d",
            self.graph.vcount(),
            self.graph.ecount(),
            len(self.passage_node_idxs),
        )


    def search(self, request: SearchGraphRequest) -> SearchGraphResult:
        retrieval_query = self._build_retrieval_query(request)
        filter_query = self._build_filter_query(request)

        candidate_triples = self.triple_index.retrieve_candidate_triples(
            query=retrieval_query,
            top_k=request.top_k_triples or self.config.top_k_candidate_triples,
        )

        filtered_triples = self._filter_triples(
            question=filter_query,
            candidate_triples=candidate_triples,
        )

        seed_info, reset_vector, seed_stats = self._build_reset_vector(
            request=request,
            retrieval_query=retrieval_query,
            candidate_triples=candidate_triples,
            filtered_triples=filtered_triples,
        )

        sorted_passage_ids, sorted_scores = self.run_ppr(reset_vector)

        top_k = request.top_k_passages or self.config.top_k_passages
        retrieved_passages = self._build_retrieved_passages(
            sorted_passage_ids=sorted_passage_ids,
            sorted_scores=sorted_scores,
            top_k=top_k,
        )

        debug_info = PPRSearchDebugInfo(
            retrieval_query=retrieval_query,
            filter_query=filter_query,
            num_candidate_triples=len(candidate_triples),
            num_filtered_triples=len(filtered_triples),
            num_entity_seeds=len(seed_info.entity_seeds),
            num_passage_seeds=len(seed_info.passage_seeds),
            num_dense_passage_seeds=seed_stats["num_dense_passage_seeds"],
            num_selected_triple_passage_seeds=seed_stats["num_selected_triple_passage_seeds"],
            reset_mass=float(np.sum(reset_vector)),
            damping=self.config.damping,
            passage_node_weight=self.config.passage_node_weight,
            selected_triple_passage_weight=self.config.selected_triple_passage_weight,
        )

        result = SearchGraphResult(
            request=request,
            passages=retrieved_passages,
            candidate_triples=candidate_triples,
            filtered_triples=filtered_triples,
            seed_info=seed_info,
            metadata={
                "debug": asdict(debug_info),
            },
        )

        if self.config.save_debug:
            self._save_debug_result(result)

        return result


    def _build_retrieval_query(self, request: SearchGraphRequest) -> str:
        pieces: List[str] = []

        if request.question:
            pieces.append(f"Question: {request.question}")

        if request.search_focus and request.search_focus.strip() != request.question.strip():
            pieces.append(f"Search focus: {request.search_focus}")

        if request.seed_entities:
            pieces.append(
                "Seed entities: "
                + ", ".join(_dedupe_strings(request.seed_entities))
            )

        if request.relation_hints:
            pieces.append(
                "Relation hints: "
                + ", ".join(_dedupe_strings(request.relation_hints))
            )

        if not pieces:
            return request.build_search_query()

        return "\n".join(pieces)

    def _build_filter_query(self, request: SearchGraphRequest) -> str:
        pieces: List[str] = [f"Question: {request.question}"]

        if request.search_focus:
            pieces.append(f"Current search focus: {request.search_focus}")

        if request.seed_entities:
            pieces.append(
                "Seed entities: "
                + ", ".join(_dedupe_strings(request.seed_entities))
            )

        if request.relation_hints:
            pieces.append(
                "Relation hints: "
                + ", ".join(_dedupe_strings(request.relation_hints))
            )

        if request.evidence_so_far:
            compact_evidence = [
                _truncate(x, max_chars=350)
                for x in request.evidence_so_far[:4]
            ]
            pieces.append(
                "Brief evidence so far:\n"
                + "\n".join(f"- {x}" for x in compact_evidence)
            )

        return "\n".join(pieces)


    def _filter_triples(
        self,
        question: str,
        candidate_triples: Sequence[CandidateTriple],
    ) -> List[FilteredTriple]:
        if not candidate_triples:
            return []

        if not self.config.enable_triple_filter or self.triple_filter is None:
            max_n = self.config.max_filtered_triples or len(candidate_triples)
            return [
                _as_filtered_from_candidate(candidate, rank=i + 1)
                for i, candidate in enumerate(candidate_triples[:max_n])
            ]

        filter_result = self.triple_filter.filter(
            question=question,
            candidates=candidate_triples,
            len_after_filter=self.config.max_filtered_triples,
        )

        filtered = filter_result.filtered_triples

        if not filtered and self.config.fallback_to_candidates_if_filter_empty:
            max_n = self.config.max_filtered_triples or len(candidate_triples)
            filtered = [
                _as_filtered_from_candidate(candidate, rank=i + 1)
                for i, candidate in enumerate(candidate_triples[:max_n])
            ]

        return filtered


    def _build_reset_vector(
        self,
        request: SearchGraphRequest,
        retrieval_query: str,
        candidate_triples: Sequence[CandidateTriple],
        filtered_triples: Sequence[FilteredTriple],
    ) -> Tuple[PPRSeedInfo, np.ndarray, Dict[str, int]]:
        num_nodes = self.graph.vcount()

        entity_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        entity_seed_scores: Dict[str, float] = {}
        passage_seed_scores: Dict[str, float] = {}

        entity_score_accumulator: Dict[str, List[float]] = defaultdict_list()

        for filtered in filtered_triples:
            triple = filtered.triple
            subject_text, object_text = _triple_entities(triple)

            base_score = filtered.original_score
            if base_score is None:
                base_score = self.config.filtered_triple_weight

            for entity_text in [subject_text, object_text]:
                entity_id = self._resolve_entity_id(entity_text)

                if entity_id is None:
                    continue

                score = _safe_float(base_score, self.config.filtered_triple_weight)

                num_occurs = len(self.entity_to_passage_ids.get(entity_id, []))
                if num_occurs > 0:
                    score = score / float(num_occurs)

                entity_score_accumulator[entity_id].append(score)

        for entity_id, scores in entity_score_accumulator.items():
            if not scores:
                continue
            entity_seed_scores[entity_id] = float(np.mean(scores))

        for seed_text in request.seed_entities:
            entity_id = self._resolve_entity_id(seed_text)
            if entity_id is None:
                continue

            entity_seed_scores[entity_id] = entity_seed_scores.get(
                entity_id,
                0.0,
            ) + float(self.config.seed_entity_weight)

        entity_seed_scores = _top_k_dict(
            entity_seed_scores,
            self.config.linking_top_k,
        )

        for entity_id, score in entity_seed_scores.items():
            node_idx = self.node_name_to_vertex_idx.get(entity_id)
            if node_idx is not None:
                entity_weights[node_idx] = float(score)

        selected_source_passage_scores = self._source_passage_scores_from_filtered_triples(
            filtered_triples=filtered_triples,
            candidate_triples=candidate_triples,
        )
        selected_source_passage_scores = _top_k_dict(
            selected_source_passage_scores,
            self.config.max_selected_triple_passage_seeds,
        )

        for passage_id, score in selected_source_passage_scores.items():
            node_idx = self.node_name_to_vertex_idx.get(passage_id)
            if node_idx is None:
                continue

            weighted_score = (
                float(score)
                * float(self.config.selected_triple_passage_weight)
            )

            passage_weights[node_idx] += weighted_score
            passage_seed_scores[passage_id] = passage_seed_scores.get(
                passage_id,
                0.0,
            ) + weighted_score

        dense_passages = self.triple_index.retrieve_dense_passages(
            query=retrieval_query,
            top_k=self.config.dense_reset_top_k,
        )

        dense_scores = np.array([p.score for p in dense_passages], dtype=np.float32)
        dense_scores = min_max_normalize(dense_scores)

        for passage, dense_score in zip(dense_passages, dense_scores):
            node_idx = self.node_name_to_vertex_idx.get(passage.passage_id)
            if node_idx is None:
                continue

            weighted_score = float(dense_score) * float(self.config.passage_node_weight)

            passage_weights[node_idx] += weighted_score
            passage_seed_scores[passage.passage_id] = passage_seed_scores.get(
                passage.passage_id,
                0.0,
            ) + weighted_score

        reset_vector = entity_weights + passage_weights
        reset_vector = self._sanitize_reset_vector(reset_vector)

        selected_triples = [filtered.triple for filtered in filtered_triples]

        seed_info = PPRSeedInfo(
            entity_seeds=entity_seed_scores,
            passage_seeds=passage_seed_scores,
            selected_triples=selected_triples,
            dense_passage_ids=[p.passage_id for p in dense_passages],
        )

        seed_stats = {
            "num_dense_passage_seeds": len(dense_passages),
            "num_selected_triple_passage_seeds": len(selected_source_passage_scores),
        }

        return seed_info, reset_vector, seed_stats

    def _source_passage_scores_from_filtered_triples(
        self,
        filtered_triples: Sequence[FilteredTriple],
        candidate_triples: Sequence[CandidateTriple],
    ) -> Dict[str, float]:
        candidate_by_index: Dict[int, CandidateTriple] = {
            int(c.index): c for c in candidate_triples
        }

        passage_scores: Dict[str, float] = {}

        for filtered in filtered_triples:
            candidate = candidate_by_index.get(int(filtered.original_index))
            if candidate is None:
                continue

            passage_id = candidate.source_passage_id
            if not passage_id:
                continue

            score = filtered.original_score
            if score is None:
                score = candidate.score

            score = _safe_float(score, default=1.0)

            rank = getattr(filtered, "filter_rank", None)
            rank_bonus = 1.0
            if isinstance(rank, int) and rank > 0:
                rank_bonus = 1.0 / float(rank)

            final_score = score + rank_bonus

            passage_scores[passage_id] = max(
                passage_scores.get(passage_id, 0.0),
                final_score,
            )

        return passage_scores

    def _sanitize_reset_vector(self, reset_vector: np.ndarray) -> np.ndarray:
        reset_vector = np.asarray(reset_vector, dtype=np.float32)
        reset_vector = np.where(
            np.isnan(reset_vector) | (reset_vector < 0),
            0,
            reset_vector,
        )

        if float(np.sum(reset_vector)) <= 0.0:
            logger.warning(
                "PPR reset vector has zero mass. Falling back to uniform passage reset."
            )

            reset_vector = np.zeros(self.graph.vcount(), dtype=np.float32)

            if not self.passage_node_idxs:
                reset_vector[:] = 1.0
            else:
                for idx in self.passage_node_idxs:
                    reset_vector[idx] = 1.0

        return reset_vector


    def _resolve_entity_id(self, entity_text: str) -> Optional[str]:
        normalized = normalize_text(entity_text)
        if not normalized:
            return None

        if normalized in self.entity_text_to_id:
            return self.entity_text_to_id[normalized]

        entity_id = entity_id_from_text(normalized)
        if entity_id in self.node_name_to_vertex_idx:
            return entity_id

        return None


    def run_ppr(self, reset_vector: np.ndarray) -> Tuple[List[str], np.ndarray]:
        reset_vector = self._sanitize_reset_vector(reset_vector)

        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(self.graph.vcount()),
            damping=self.config.damping,
            directed=self.graph.is_directed(),
            weights="weight" if "weight" in self.graph.es.attributes() else None,
            reset=reset_vector,
            implementation="prpack",
        )

        pagerank_scores = np.asarray(pagerank_scores, dtype=np.float32)

        passage_scores = np.array(
            [pagerank_scores[idx] for idx in self.passage_node_idxs],
            dtype=np.float32,
        )

        sorted_local_indices = np.argsort(passage_scores)[::-1]
        sorted_scores = passage_scores[sorted_local_indices]

        sorted_passage_ids = [
            self.graph.vs[self.passage_node_idxs[i]]["name"]
            for i in sorted_local_indices.tolist()
        ]

        return sorted_passage_ids, sorted_scores


    def _build_retrieved_passages(
        self,
        sorted_passage_ids: Sequence[str],
        sorted_scores: Sequence[float],
        top_k: int,
    ) -> List[RetrievedPassage]:
        passages: List[RetrievedPassage] = []

        for rank, (passage_id, score) in enumerate(
            zip(sorted_passage_ids[:top_k], sorted_scores[:top_k]),
            start=1,
        ):
            text = self.passage_id_to_text.get(passage_id, "")
            metadata = self.passage_id_to_metadata.get(passage_id, {})

            passages.append(
                RetrievedPassage(
                    passage_id=passage_id,
                    text=text,
                    score=float(score),
                    rank=rank,
                    title=metadata.get("title"),
                    metadata=metadata,
                )
            )

        return passages


    def _save_debug_result(self, result: SearchGraphResult) -> None:
        debug_dir = self.config.debug_dir or os.path.join(
            self.config.index_dir,
            "ppr_debug",
        )
        os.makedirs(debug_dir, exist_ok=True)

        step_id = result.request.step_id
        filename = f"search_step_{step_id}.json"
        path = os.path.join(debug_dir, filename)

        payload = {
            "request": asdict(result.request),
            "passages": [asdict(p) for p in result.passages],
            "candidate_triples": [
                {
                    "index": c.index,
                    "score": c.score,
                    "source_passage_id": c.source_passage_id,
                    "triple": asdict(c.triple),
                }
                for c in result.candidate_triples
            ],
            "filtered_triples": [
                {
                    "original_index": f.original_index,
                    "original_score": f.original_score,
                    "filter_rank": f.filter_rank,
                    "filter_confidence": f.filter_confidence,
                    "triple": asdict(f.triple),
                }
                for f in result.filtered_triples
            ],
            "seed_info": asdict(result.seed_info) if result.seed_info else None,
            "metadata": result.metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Saved PPR debug result to %s", path)


def defaultdict_list() -> Dict[str, List[float]]:
    from collections import defaultdict
    return defaultdict(list)

def search_graph_once(
    question: str,
    graph_dir: str,
    index_dir: str,
    embedding_model: Any,
    top_k_passages: int = 5,
    seed_entities: Optional[List[str]] = None,
    relation_hints: Optional[List[str]] = None,
    search_focus: Optional[str] = None,
    enable_triple_filter: bool = False,
) -> SearchGraphResult:
    config = PPRSearchConfig(
        graph_dir=graph_dir,
        index_dir=index_dir,
        enable_triple_filter=enable_triple_filter,
        top_k_passages=top_k_passages,
    )
    engine = PPRSearchEngine(
        config=config,
        embedding_model=embedding_model,
    )
    request = SearchGraphRequest(
        question=question,
        search_focus=search_focus,
        seed_entities=seed_entities or [],
        relation_hints=relation_hints or [],
        top_k_passages=top_k_passages,
    )

    return engine.search(request)
