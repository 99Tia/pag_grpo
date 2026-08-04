"""Hybrid PPR + LLM-selected evidence fusion.

This module is the reusable package form of:

    scripts/hybrid_merge_ppr_llmselect.py

The default behavior intentionally preserves the working `keep2` fusion rule:

1. Keep the first `keep_ppr_top_n` passages from the original PPR order.
2. Fill the remaining leading positions from the LLM-selected/reranked order.
3. Append the rest of the original PPR ranking.
4. Remove duplicate passages while preserving order.

The module does not call an LLM and does not modify retrieval scores.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class EvidenceFusionConfig:
    """Configuration for hybrid PPR–LLM evidence fusion."""

    keep_ppr_top_n: int = 2
    target_top_k: int = 5

    # Use deep copies so one GRPO trajectory cannot mutate another.
    copy_passages: bool = True

    def validate(self) -> None:
        if self.keep_ppr_top_n < 0:
            raise ValueError("keep_ppr_top_n cannot be negative.")

        if self.target_top_k <= 0:
            raise ValueError("target_top_k must be greater than zero.")

        if self.keep_ppr_top_n > self.target_top_k:
            raise ValueError(
                "keep_ppr_top_n cannot be greater than target_top_k."
            )


@dataclass
class EvidenceFusionResult:
    """Result returned by one hybrid-fusion call."""

    passages: List[Dict[str, Any]]

    keep_ppr_top_n: int
    target_top_k: int

    num_base_passages: int
    num_llm_passages: int
    num_output_passages: int

    ppr_kept_ids: List[str]
    llm_added_ids: List[str]
    final_top_k_ids: List[str]

    metadata: Dict[str, Any]


def to_plain(value: Any) -> Any:
    """Convert framework objects and dataclasses to plain Python values."""

    if value is None:
        return None

    if is_dataclass(value):
        return {
            key: to_plain(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): to_plain(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "__dict__"):
        return {
            str(key): to_plain(item)
            for key, item in vars(value).items()
        }

    return str(value)


def normalize_passage(passage: Any) -> Optional[Dict[str, Any]]:
    """Convert a passage object into a dictionary without changing its fields."""

    plain = to_plain(passage)

    if isinstance(plain, str):
        text = plain.strip()

        if not text:
            return None

        return {
            "text": text,
            "metadata": {},
        }

    if not isinstance(plain, dict):
        return None

    return plain


def passage_aliases(passage: Dict[str, Any]) -> Set[str]:
    """Collect known IDs that may refer to the same passage."""

    identifiers: Set[str] = set()

    for key in [
        "id",
        "passage_id",
        "node_id",
        "idx",
        "fallback_idx",
    ]:
        value = passage.get(key)

        if value is not None:
            identifiers.add(str(value))

    metadata = passage.get("metadata") or {}

    if not isinstance(metadata, dict):
        metadata = {}

    for key in [
        "fallback_idx",
        "idx",
        "passage_idx",
        "source_idx",
        "corpus_idx",
        "passage_id",
        "node_id",
    ]:
        value = metadata.get(key)

        if value is not None:
            identifiers.add(str(value))

    return identifiers


def passage_key(passage: Dict[str, Any]) -> Tuple[str, ...]:
    """Build a stable deduplication key matching the original script."""

    identifiers = sorted(passage_aliases(passage))

    if identifiers:
        return tuple(identifiers)

    return (
        json.dumps(
            passage,
            sort_keys=True,
            ensure_ascii=False,
        ),
    )


def primary_passage_id(passage: Dict[str, Any]) -> str:
    """Return one readable identifier for logs and metadata."""

    for key in [
        "passage_id",
        "id",
        "idx",
        "node_id",
        "fallback_idx",
    ]:
        value = passage.get(key)

        if value is not None:
            return str(value)

    metadata = passage.get("metadata") or {}

    if isinstance(metadata, dict):
        for key in [
            "passage_id",
            "idx",
            "corpus_idx",
            "source_idx",
            "node_id",
            "fallback_idx",
        ]:
            value = metadata.get(key)

            if value is not None:
                return str(value)

    return "unknown"


def add_unique(
    output: List[Dict[str, Any]],
    passage: Dict[str, Any],
    seen: Set[Tuple[str, ...]],
) -> bool:
    """Append a passage when it has not already been added."""

    key = passage_key(passage)

    if key in seen:
        return False

    seen.add(key)
    output.append(passage)

    return True


def add_fusion_metadata(
    passages: Sequence[Dict[str, Any]],
    ppr_kept_keys: Set[Tuple[str, ...]],
    llm_added_keys: Set[Tuple[str, ...]],
    target_top_k: int,
) -> None:
    """Attach analysis metadata without affecting ranking."""

    for final_rank, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}
            passage["metadata"] = metadata

        key = passage_key(passage)

        metadata["hybrid_ppr_llmselect_final_rank"] = final_rank
        metadata["hybrid_ppr_llmselect_in_top_k"] = (
            final_rank <= target_top_k
        )
        metadata["hybrid_ppr_llmselect_source"] = (
            "ppr_fixed"
            if key in ppr_kept_keys
            else "llm_selected"
            if key in llm_added_keys
            else "ppr_tail"
        )


class HybridEvidenceFuser:
    """Apply the fixed hybrid PPR–LLM evidence-fusion rule."""

    def __init__(self, config: EvidenceFusionConfig):
        config.validate()
        self.config = config

    def merge(
        self,
        base_passages: Sequence[Any],
        llm_passages: Sequence[Any],
    ) -> EvidenceFusionResult:
        """Fuse original PPR evidence with LLM-selected evidence.

        Parameters
        ----------
        base_passages:
            Original PPR-ranked evidence list.

        llm_passages:
            Evidence list reordered by EvidenceSelectorV2.

        Returns
        -------
        EvidenceFusionResult
            The complete fused ranking. The first `target_top_k` passages are
            the final evidence set used by the answer reader and reward.
        """

        normalized_base: List[Dict[str, Any]] = []
        normalized_llm: List[Dict[str, Any]] = []

        for passage in base_passages:
            normalized = normalize_passage(passage)

            if normalized is not None:
                normalized_base.append(
                    deepcopy(normalized)
                    if self.config.copy_passages
                    else normalized
                )

        for passage in llm_passages:
            normalized = normalize_passage(passage)

            if normalized is not None:
                normalized_llm.append(
                    deepcopy(normalized)
                    if self.config.copy_passages
                    else normalized
                )

        output: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()

        ppr_kept_keys: Set[Tuple[str, ...]] = set()
        llm_added_keys: Set[Tuple[str, ...]] = set()

        ppr_kept_ids: List[str] = []
        llm_added_ids: List[str] = []

        # 1. Preserve the strongest fixed PPR prefix.
        for passage in normalized_base[
            : self.config.keep_ppr_top_n
        ]:
            if add_unique(output, passage, seen):
                key = passage_key(passage)
                ppr_kept_keys.add(key)
                ppr_kept_ids.append(
                    primary_passage_id(passage)
                )

        # 2. Fill the leading evidence positions from the LLM order.
        for passage in normalized_llm:
            if len(output) >= self.config.target_top_k:
                break

            if add_unique(output, passage, seen):
                key = passage_key(passage)
                llm_added_keys.add(key)
                llm_added_ids.append(
                    primary_passage_id(passage)
                )

        # 3. Append the remaining original PPR ranking.
        #
        # This intentionally follows the original script: the final complete
        # list cannot exceed the number of passages in the base PPR list.
        for passage in normalized_base:
            if len(output) >= len(normalized_base):
                break

            add_unique(output, passage, seen)

        add_fusion_metadata(
            passages=output,
            ppr_kept_keys=ppr_kept_keys,
            llm_added_keys=llm_added_keys,
            target_top_k=self.config.target_top_k,
        )

        final_top_k_ids = [
            primary_passage_id(passage)
            for passage in output[
                : self.config.target_top_k
            ]
        ]

        metadata = {
            "keep_ppr_top_n": self.config.keep_ppr_top_n,
            "target_top_k": self.config.target_top_k,
            "num_base_passages": len(normalized_base),
            "num_llm_passages": len(normalized_llm),
            "num_output_passages": len(output),
            "ppr_kept_ids": ppr_kept_ids,
            "llm_added_ids": llm_added_ids,
            "final_top_k_ids": final_top_k_ids,
        }

        return EvidenceFusionResult(
            passages=output,
            keep_ppr_top_n=self.config.keep_ppr_top_n,
            target_top_k=self.config.target_top_k,
            num_base_passages=len(normalized_base),
            num_llm_passages=len(normalized_llm),
            num_output_passages=len(output),
            ppr_kept_ids=ppr_kept_ids,
            llm_added_ids=llm_added_ids,
            final_top_k_ids=final_top_k_ids,
            metadata=metadata,
        )

    def merge_rows(
        self,
        base_row: Dict[str, Any],
        llm_row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compatibility helper for the original JSONL script workflow."""

        new_row = deepcopy(base_row)

        base_passages = (
            base_row.get("evidence_passages")
            or base_row.get("retrieved_passages")
            or base_row.get("passages")
            or []
        )

        llm_passages = (
            llm_row.get("evidence_passages")
            or llm_row.get("retrieved_passages")
            or llm_row.get("passages")
            or []
        )

        result = self.merge(
            base_passages=base_passages,
            llm_passages=llm_passages,
        )

        if "evidence_passages" in new_row:
            new_row["evidence_passages"] = result.passages

        elif "retrieved_passages" in new_row:
            new_row["retrieved_passages"] = result.passages

        elif "passages" in new_row:
            new_row["passages"] = result.passages

        else:
            new_row["evidence_passages"] = result.passages

        metadata = new_row.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}
            new_row["metadata"] = metadata

        metadata["hybrid_ppr_llmselect"] = dict(
            result.metadata
        )

        return new_row


def merge_evidence(
    base_passages: Sequence[Any],
    llm_passages: Sequence[Any],
    keep_ppr_top_n: int = 2,
    target_top_k: int = 5,
) -> EvidenceFusionResult:
    """Functional convenience wrapper."""

    fuser = HybridEvidenceFuser(
        EvidenceFusionConfig(
            keep_ppr_top_n=keep_ppr_top_n,
            target_top_k=target_top_k,
        )
    )

    return fuser.merge(
        base_passages=base_passages,
        llm_passages=llm_passages,
    )
