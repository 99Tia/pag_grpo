"""Triple/passage/entity indexing for the framework. This module builds disk-cached embedding indexes for:
    chunk  -> passage texts
    entity -> entity/phrase texts
    fact   -> extracted triples/facts"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .embedding_store import EmbeddingStore, make_default_embedding_stores
from .schema import (
    CandidateTriple,
    RetrievedPassage,
    Triple,
    normalize_text,
    triple_to_text,
)

logger = logging.getLogger(__name__)


@dataclass
class TripleIndexConfig:
    index_dir: str
    graph_metadata_path: str

    batch_size: int = 32
    normalize_embeddings: bool = True

    query_to_fact_instruction: Optional[str] = None
    query_to_passage_instruction: Optional[str] = None

    index_metadata_filename: str = "triple_index_metadata.json"


@dataclass
class TripleIndexBuildResult:
    index_dir: str
    num_passages: int
    num_entities: int
    num_triples: int
    metadata_path: str


def min_max_normalize(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)

    if scores.size == 0:
        return scores

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score - min_score < eps:
        return np.zeros_like(scores, dtype=np.float32)

    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)


def _safe_first(values: Any) -> Optional[str]:
    if isinstance(values, list) and values:
        return values[0]
    if isinstance(values, str):
        return values
    return None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _make_triple_from_record(record: Dict[str, Any]) -> Triple:
    source_passage_ids = record.get("source_passage_ids", [])
    return Triple(
        subject=record["subject"],
        predicate=record["predicate"],
        object=record["object"],
        source_passage_id=_safe_first(source_passage_ids),
        triple_id=record.get("triple_id"),
        metadata={
            "source_passage_ids": source_passage_ids,
            **dict(record.get("metadata", {})),
        },
    )


class TripleIndex:
    def __init__(
        self,
        config: TripleIndexConfig,
        embedding_model: Any = None,
    ):
        self.config = config
        self.embedding_model = embedding_model

        os.makedirs(self.config.index_dir, exist_ok=True)

        self.index_metadata_path = os.path.join(
            self.config.index_dir,
            self.config.index_metadata_filename,
        )

        self.stores: Dict[str, EmbeddingStore] = make_default_embedding_stores(
            root_dir=self.config.index_dir,
            embedding_model=embedding_model,
            batch_size=self.config.batch_size,
            normalize=self.config.normalize_embeddings,
        )

        self.chunk_store = self.stores["chunk"]
        self.entity_store = self.stores["entity"]
        self.fact_store = self.stores["fact"]

        self.graph_metadata: Dict[str, Any] = {}
        self.index_metadata: Dict[str, Any] = {}

        self.passage_records: List[Dict[str, Any]] = []
        self.entity_records: List[Dict[str, Any]] = []
        self.triple_records: List[Dict[str, Any]] = []

        self.passage_store_ids: List[str] = []
        self.entity_store_ids: List[str] = []
        self.fact_store_ids: List[str] = []

        self.passage_id_to_store_id: Dict[str, str] = {}
        self.store_id_to_passage_id: Dict[str, str] = {}

        self.triple_id_to_store_id: Dict[str, str] = {}
        self.store_id_to_triple_id: Dict[str, str] = {}

        self._maybe_load_index_metadata()


    def build(self, force: bool = False) -> TripleIndexBuildResult:
        if os.path.exists(self.index_metadata_path) and not force:
            logger.info("Triple index metadata exists. Loading: %s", self.index_metadata_path)
            self.load()
            return TripleIndexBuildResult(
                index_dir=self.config.index_dir,
                num_passages=len(self.passage_records),
                num_entities=len(self.entity_records),
                num_triples=len(self.triple_records),
                metadata_path=self.index_metadata_path,
            )

        self.graph_metadata = _load_json(self.config.graph_metadata_path)

        self._prepare_passage_records()
        self._prepare_entity_records()
        self._prepare_triple_records()

        logger.info("Indexing passages: %d", len(self.passage_records))
        passage_texts = [r["text"] for r in self.passage_records]
        passage_metadata = [
            {
                "passage_id": r["passage_id"],
                **dict(r.get("metadata", {})),
            }
            for r in self.passage_records
        ]
        self.passage_store_ids = self.chunk_store.insert_strings(
            passage_texts,
            metadata=passage_metadata,
        )

        logger.info("Indexing entities: %d", len(self.entity_records))
        entity_texts = [r["text"] for r in self.entity_records]
        entity_metadata = [
            {
                "entity_id": r["entity_id"],
            }
            for r in self.entity_records
        ]
        self.entity_store_ids = self.entity_store.insert_strings(
            entity_texts,
            metadata=entity_metadata,
        )

        logger.info("Indexing triples/facts: %d", len(self.triple_records))
        fact_texts = [r["text"] for r in self.triple_records]
        fact_metadata = [
            {
                "triple_id": r["triple_id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "source_passage_ids": r.get("source_passage_ids", []),
            }
            for r in self.triple_records
        ]
        self.fact_store_ids = self.fact_store.insert_strings(
            fact_texts,
            metadata=fact_metadata,
        )

        self._build_mappings()
        self._save_index_metadata()

        return TripleIndexBuildResult(
            index_dir=self.config.index_dir,
            num_passages=len(self.passage_records),
            num_entities=len(self.entity_records),
            num_triples=len(self.triple_records),
            metadata_path=self.index_metadata_path,
        )

    def load(self) -> None:
        if not os.path.exists(self.index_metadata_path):
            raise FileNotFoundError(
                f"Index metadata not found. Build the index first: {self.index_metadata_path}"
            )

        self.index_metadata = _load_json(self.index_metadata_path)

        self.passage_records = list(self.index_metadata.get("passage_records", []))
        self.entity_records = list(self.index_metadata.get("entity_records", []))
        self.triple_records = list(self.index_metadata.get("triple_records", []))

        self.passage_store_ids = list(self.index_metadata.get("passage_store_ids", []))
        self.entity_store_ids = list(self.index_metadata.get("entity_store_ids", []))
        self.fact_store_ids = list(self.index_metadata.get("fact_store_ids", []))

        self.passage_id_to_store_id = dict(
            self.index_metadata.get("passage_id_to_store_id", {})
        )
        self.store_id_to_passage_id = dict(
            self.index_metadata.get("store_id_to_passage_id", {})
        )

        self.triple_id_to_store_id = dict(
            self.index_metadata.get("triple_id_to_store_id", {})
        )
        self.store_id_to_triple_id = dict(
            self.index_metadata.get("store_id_to_triple_id", {})
        )

        self.graph_metadata = _load_json(self.config.graph_metadata_path)

        logger.info(
            "Loaded triple index: passages=%d, entities=%d, triples=%d",
            len(self.passage_records),
            len(self.entity_records),
            len(self.triple_records),
        )

    def _maybe_load_index_metadata(self) -> None:
        if os.path.exists(self.index_metadata_path):
            try:
                self.load()
            except Exception as exc:
                logger.warning("Could not auto-load triple index metadata: %s", exc)


    def _prepare_passage_records(self) -> None:
        passage_id_to_text = self.graph_metadata.get("passage_id_to_text", {})
        passage_id_to_metadata = self.graph_metadata.get("passage_id_to_metadata", {})

        self.passage_records = []

        for passage_id, text in passage_id_to_text.items():
            self.passage_records.append(
                {
                    "passage_id": passage_id,
                    "text": text,
                    "metadata": passage_id_to_metadata.get(passage_id, {}),
                }
            )

    def _prepare_entity_records(self) -> None:
        entity_id_to_text = self.graph_metadata.get("entity_id_to_text", {})

        self.entity_records = []

        for entity_id, text in entity_id_to_text.items():
            self.entity_records.append(
                {
                    "entity_id": entity_id,
                    "text": normalize_text(text),
                }
            )

    def _prepare_triple_records(self) -> None:
        raw_records = self.graph_metadata.get("triple_records", [])

        self.triple_records = []

        for record in raw_records:
            subject = normalize_text(record["subject"])
            predicate = normalize_text(record["predicate"])
            object_ = normalize_text(record["object"])

            text = record.get("text") or triple_to_text(subject, predicate, object_)

            self.triple_records.append(
                {
                    "triple_id": record["triple_id"],
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "text": text,
                    "source_passage_ids": list(record.get("source_passage_ids", [])),
                    "metadata": dict(record.get("metadata", {})),
                }
            )

    def _build_mappings(self) -> None:
        self.passage_id_to_store_id = {}
        self.store_id_to_passage_id = {}

        for record, store_id in zip(self.passage_records, self.passage_store_ids):
            passage_id = record["passage_id"]
            self.passage_id_to_store_id[passage_id] = store_id
            self.store_id_to_passage_id[store_id] = passage_id

        self.triple_id_to_store_id = {}
        self.store_id_to_triple_id = {}

        for record, store_id in zip(self.triple_records, self.fact_store_ids):
            triple_id = record["triple_id"]
            self.triple_id_to_store_id[triple_id] = store_id
            self.store_id_to_triple_id[store_id] = triple_id

    def _save_index_metadata(self) -> None:
        self.index_metadata = {
            "config": asdict(self.config),
            "passage_records": self.passage_records,
            "entity_records": self.entity_records,
            "triple_records": self.triple_records,
            "passage_store_ids": self.passage_store_ids,
            "entity_store_ids": self.entity_store_ids,
            "fact_store_ids": self.fact_store_ids,
            "passage_id_to_store_id": self.passage_id_to_store_id,
            "store_id_to_passage_id": self.store_id_to_passage_id,
            "triple_id_to_store_id": self.triple_id_to_store_id,
            "store_id_to_triple_id": self.store_id_to_triple_id,
        }

        _save_json(self.index_metadata_path, self.index_metadata)

        logger.info("Saved triple index metadata to %s", self.index_metadata_path)


    def _require_loaded(self) -> None:
        if not self.triple_records:
            self.load()

    def _encode_query(
        self,
        query: str,
        store: EmbeddingStore,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        encode_kwargs: Dict[str, Any] = {}

        if instruction:
            encode_kwargs["instruction"] = instruction

        emb = store.embedding_model.encode(
            [query],
            batch_size=1,
            normalize=True,
            **encode_kwargs,
        )

        return emb[0].astype(np.float32)

    def _score_embeddings(
        self,
        query_embedding: np.ndarray,
        matrix: np.ndarray,
        normalize_scores: bool = True,
    ) -> np.ndarray:
        if matrix.size == 0:
            return np.array([], dtype=np.float32)

        scores = np.dot(matrix, query_embedding.T).astype(np.float32)

        if normalize_scores:
            scores = min_max_normalize(scores)

        return scores


    def get_fact_scores(
        self,
        query: str,
        normalize_scores: bool = True,
    ) -> np.ndarray:
        """Compute query-to-triple similarity scores."""
        self._require_loaded()

        query_embedding = self._encode_query(
            query,
            store=self.fact_store,
            instruction=self.config.query_to_fact_instruction,
        )

        fact_embeddings = self.fact_store.get_embeddings(self.fact_store_ids)

        return self._score_embeddings(
            query_embedding=query_embedding,
            matrix=fact_embeddings,
            normalize_scores=normalize_scores,
        )

    def retrieve_candidate_triples(
        self,
        query: str,
        top_k: int = 50,
        normalize_scores: bool = True,
    ) -> List[CandidateTriple]:
        """Retrieve candidate triples before LLM filtering."""
        self._require_loaded()

        scores = self.get_fact_scores(
            query=query,
            normalize_scores=normalize_scores,
        )

        if scores.size == 0:
            return []

        top_k = min(top_k, len(scores))
        sorted_indices = np.argsort(scores)[::-1][:top_k].tolist()

        candidates: List[CandidateTriple] = []

        for rank_idx in sorted_indices:
            record = self.triple_records[rank_idx]
            triple = _make_triple_from_record(record)

            candidates.append(
                CandidateTriple(
                    triple=triple,
                    index=rank_idx,
                    score=float(scores[rank_idx]),
                    source_passage_id=triple.source_passage_id,
                )
            )

        return candidates


    def get_passage_scores(
        self,
        query: str,
        normalize_scores: bool = True,
    ) -> np.ndarray:
        """Compute query-to-passage similarity scores."""
        self._require_loaded()

        query_embedding = self._encode_query(
            query,
            store=self.chunk_store,
            instruction=self.config.query_to_passage_instruction,
        )

        passage_embeddings = self.chunk_store.get_embeddings(self.passage_store_ids)

        return self._score_embeddings(
            query_embedding=query_embedding,
            matrix=passage_embeddings,
            normalize_scores=normalize_scores,
        )

    def retrieve_dense_passages(
        self,
        query: str,
        top_k: int = 50,
        normalize_scores: bool = True,
    ) -> List[RetrievedPassage]:
        self._require_loaded()

        scores = self.get_passage_scores(
            query=query,
            normalize_scores=normalize_scores,
        )

        if scores.size == 0:
            return []

        top_k = min(top_k, len(scores))
        sorted_indices = np.argsort(scores)[::-1][:top_k].tolist()

        passages: List[RetrievedPassage] = []

        for rank, idx in enumerate(sorted_indices, start=1):
            record = self.passage_records[idx]
            passage_id = record["passage_id"]

            passages.append(
                RetrievedPassage(
                    passage_id=passage_id,
                    text=record["text"],
                    score=float(scores[idx]),
                    rank=rank,
                    title=record.get("metadata", {}).get("title"),
                    metadata=record.get("metadata", {}),
                )
            )

        return passages


    def get_passage_text(self, passage_id: str) -> str:
        self._require_loaded()

        for record in self.passage_records:
            if record["passage_id"] == passage_id:
                return record["text"]

        raise KeyError(f"Unknown passage_id: {passage_id}")

    def get_passage_record(self, passage_id: str) -> Dict[str, Any]:
        self._require_loaded()

        for record in self.passage_records:
            if record["passage_id"] == passage_id:
                return record

        raise KeyError(f"Unknown passage_id: {passage_id}")

    def get_triple_record_by_index(self, index: int) -> Dict[str, Any]:
        self._require_loaded()
        return self.triple_records[index]

    def get_triple_by_index(self, index: int) -> Triple:
        self._require_loaded()
        return _make_triple_from_record(self.triple_records[index])

    def num_passages(self) -> int:
        self._require_loaded()
        return len(self.passage_records)

    def num_entities(self) -> int:
        self._require_loaded()
        return len(self.entity_records)

    def num_triples(self) -> int:
        self._require_loaded()
        return len(self.triple_records)