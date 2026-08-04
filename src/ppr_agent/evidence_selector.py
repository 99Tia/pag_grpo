from __future__ import annotations
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from openai import OpenAI


@dataclass
class EvidenceSelectorConfig:
    base_url: str
    model_name: str

    api_key_env: str = "OPENAI_API_KEY"

    top_pool: int = 15
    select_k: int = 5
    max_passage_chars: int = 900
    max_triples: int = 30

    temperature: float = 0.0
    max_tokens: int = 700
    retries: int = 3
    sleep_base: float = 1.0
    fallback_to_original_order: bool = True

    def validate(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must be provided.")
        if not self.model_name:
            raise ValueError("model_name must be provided.")
        if self.top_pool <= 0:
            raise ValueError("top_pool must be greater than zero.")
        if self.select_k <= 0:
            raise ValueError("select_k must be greater than zero.")
        if self.max_passage_chars <= 0:
            raise ValueError("max_passage_chars must be greater than zero.")
        if self.max_triples < 0:
            raise ValueError("max_triples cannot be negative.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        if self.retries <= 0:
            raise ValueError("retries must be greater than zero.")


@dataclass
class EvidenceSelectorResult:
    question: str
    passages: List[Dict[str, Any]]
    selected_indices: List[int]

    inferred_hops: List[Any]
    passage_roles: List[Any]
    chain_reason: str
    missing_evidence_warning: str

    skipped: bool = False
    fallback_used: bool = False
    error: Optional[str] = None
    raw_response: Optional[str] = None

    top_pool: int = 0
    select_k: int = 0

    def metadata(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "selected_indices": list(self.selected_indices),
            "inferred_hops": list(self.inferred_hops),
            "passage_roles": list(self.passage_roles),
            "chain_reason": self.chain_reason,
            "missing_evidence_warning": self.missing_evidence_warning,
            "top_pool": self.top_pool,
            "select_k": self.select_k,
            "skipped": self.skipped,
            "fallback_used": self.fallback_used,
        }

        if self.error is not None:
            payload["error"] = self.error

        return payload


def to_plain(value: Any) -> Any:
    if value is None:
        return None

    if is_dataclass(value):
        return {key: to_plain(item) for key, item in asdict(value).items()}

    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "__dict__"):
        return {str(key): to_plain(item) for key, item in vars(value).items()}

    return str(value)


def normalize_passage(passage: Any) -> Optional[Dict[str, Any]]:
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


def passage_text(passage: Dict[str, Any]) -> str:
    return str(
        passage.get("text")
        or passage.get("passage_text")
        or passage.get("paragraph_text")
        or passage.get("content")
        or ""
    )


def passage_title(passage: Dict[str, Any]) -> str:
    metadata = passage.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    return str(
        passage.get("title")
        or metadata.get("title")
        or ""
    )


def passage_score(passage: Dict[str, Any]) -> str:
    metadata = passage.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    score = passage.get("score")

    if score is None:
        score = metadata.get("score")

    if score is None:
        score = metadata.get("ppr_score")

    if score is None:
        return ""

    return str(score)


def passage_aliases(passage: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()

    for key in ["id", "passage_id", "node_id", "idx", "fallback_idx"]:
        if passage.get(key) is not None:
            ids.add(str(passage.get(key)))

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
        if metadata.get(key) is not None:
            ids.add(str(metadata.get(key)))

    return ids


def alias_key(passage: Dict[str, Any]) -> str:
    ids = sorted(passage_aliases(passage))

    if ids:
        return "||".join(ids)

    return json.dumps(passage, sort_keys=True, ensure_ascii=False)


def triple_to_text(triple: Any) -> str:
    plain = to_plain(triple)

    if not isinstance(plain, dict):
        return str(plain)

    inner = plain.get("triple", plain)
    if not isinstance(inner, dict):
        return str(inner)

    if (
        inner.get("subject")
        and (inner.get("predicate") or inner.get("relation"))
        and inner.get("object")
    ):
        predicate = inner.get("predicate") or inner.get("relation")
        return f"{inner.get('subject')} | {predicate} | {inner.get('object')}"

    if (
        inner.get("head")
        and (inner.get("predicate") or inner.get("relation"))
        and inner.get("tail")
    ):
        predicate = inner.get("predicate") or inner.get("relation")
        return f"{inner.get('head')} | {predicate} | {inner.get('tail')}"

    parts: List[str] = []

    for key in [
        "subject",
        "predicate",
        "object",
        "relation",
        "head",
        "tail",
        "text",
        "triple_text",
    ]:
        if inner.get(key):
            parts.append(str(inner[key]))

    return " | ".join(parts)


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()

    cleaned = re.sub(
        r"^```(?:json)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if match is None:
        raise ValueError("No JSON object found in LLM output.")

    parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON must be an object.")

    return parsed


def normalize_indices(
    indices: Any,
    top_pool: int,
    select_k: int,
) -> List[int]:
    if not isinstance(indices, list):
        return []

    output: List[int] = []
    seen: Set[int] = set()

    for value in indices:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue

        if 1 <= index <= top_pool and index not in seen:
            seen.add(index)
            output.append(index)

        if len(output) >= select_k:
            break

    return output


def fill_missing_indices(
    selected_indices: Sequence[int],
    top_pool: int,
    select_k: int,
) -> List[int]:
    output = list(selected_indices)
    seen = set(output)

    for index in range(1, top_pool + 1):
        if len(output) >= select_k:
            break

        if index in seen:
            continue

        output.append(index)
        seen.add(index)

    return output


def build_prompt(
    question: str,
    passages: Sequence[Dict[str, Any]],
    filtered_triples: Sequence[Any],
    top_pool: int,
    select_k: int,
    max_passage_chars: int,
    max_triples: int,
) -> str:

    triple_lines: List[str] = []

    for index, triple in enumerate(filtered_triples[:max_triples], start=1):
        text = triple_to_text(triple)

        if text:
            triple_lines.append(f"{index}. {text[:300]}")

    passage_lines: List[str] = []

    for index, passage in enumerate(passages[:top_pool], start=1):
        text = passage_text(passage).replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()

        title = passage_title(passage)
        score = passage_score(passage)
        ids = sorted(passage_aliases(passage))
        alias_text = ", ".join(ids[:4]) if ids else ""

        passage_lines.append(
            f"[{index}]\n"
            f"title: {title}\n"
            f"score: {score}\n"
            f"ids: {alias_text}\n"
            f"text: {text[:max_passage_chars]}"
        )

    triples_text = "\n".join(triple_lines) if triple_lines else "None"
    passages_text = "\n".join(passage_lines)

    return f"""You are a chain-aware evidence selector for multi-hop question answering.

You must select exactly {select_k} passages from the candidate list.

Important:
- Do NOT answer the question.
- Do NOT use outside knowledge.
- Use only the provided candidate passages and filtered triples.
- Your goal is not to pick five individually similar passages.
- Your goal is to pick a small evidence set that together supports the full reasoning chain.

Selection rules:
1. First infer the likely reasoning hops required by the question.
2. Prefer passages that connect entities across hops, not just passages that mention the final answer.
3. Prefer a complete chain: bridge evidence + intermediate entity evidence + final answer evidence.
4. Avoid redundant passages unless redundancy is needed to connect the chain.
5. If a passage is highly ranked by PPR but does not support any needed hop, do not select it only because of the score.
6. If a lower-ranked passage completes a missing hop, promote it.
7. Return selected passage numbers in the order they should be read for reasoning.

Question:
{question}

Filtered triples:
{triples_text}

Candidate passages:
{passages_text}

Return only valid JSON with this schema:
{{
  "inferred_hops": [
    "hop 1 description",
    "hop 2 description"
  ],
  "passage_roles": [
    {{"index": 1, "role": "bridge/final/redundant/irrelevant", "supports_hop": "short explanation"}}
  ],
  "selected_indices": [1, 2, 3, 4, 5],
  "chain_reason": "short explanation of why these passages cover the full chain",
  "missing_evidence_warning": "None, or short warning if the provided candidates do not fully cover the chain"
}}
"""


def reorder_by_selection(
    passages: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    top_pool: int,
) -> List[Dict[str, Any]]:
    pool = list(passages[:top_pool])
    tail = list(passages[top_pool:])

    output: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for index in selected_indices:
        zero_based = index - 1

        if 0 <= zero_based < len(pool):
            passage = pool[zero_based]
            key = alias_key(passage)

            if key not in seen:
                seen.add(key)
                output.append(passage)

    for passage in pool:
        key = alias_key(passage)

        if key not in seen:
            seen.add(key)
            output.append(passage)

    for passage in tail:
        key = alias_key(passage)

        if key not in seen:
            seen.add(key)
            output.append(passage)

    return output


def add_rank_metadata(
    original_passages: Sequence[Dict[str, Any]],
    reordered_passages: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    top_pool: int,
) -> None:
    original_rank_by_key: Dict[str, int] = {
        alias_key(passage): rank
        for rank, passage in enumerate(original_passages, start=1)
    }

    selected_original_ranks = set(selected_indices)
    num_selected = len(selected_indices)

    for new_rank, passage in enumerate(reordered_passages[:top_pool], start=1):
        metadata = passage.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}
            passage["metadata"] = metadata

        original_rank = original_rank_by_key.get(alias_key(passage))

        metadata["llm_selector_v2_new_rank"] = new_rank
        metadata["llm_selector_v2_original_rank"] = original_rank
        metadata["llm_selector_v2_selected"] = new_rank <= num_selected

        if original_rank is not None:
            metadata["llm_selector_v2_selected_by_original_rank"] = (
                original_rank in selected_original_ranks
            )


class EvidenceSelectorV2:
    def __init__(
        self,
        config: EvidenceSelectorConfig,
        client: Optional[OpenAI] = None,
    ):
        config.validate()
        self.config = config

        if client is None:
            api_key = os.environ.get(config.api_key_env, "dummy")
            client = OpenAI(
                base_url=config.base_url,
                api_key=api_key,
            )

        self.client = client

    def _call_llm(self, prompt: str) -> tuple[Dict[str, Any], str]:
        last_error: Optional[Exception] = None

        for attempt in range(self.config.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only valid JSON. "
                                "You are selecting chain-complete evidence "
                                "passages for multi-hop QA."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                raw_response = response.choices[0].message.content or ""
                return extract_json(raw_response), raw_response

            except Exception as exc:
                last_error = exc

                if attempt + 1 < self.config.retries:
                    time.sleep(self.config.sleep_base + attempt)

        raise RuntimeError(
            f"LLM failed after {self.config.retries} retries: {last_error}"
        )

    def select(
        self,
        question: str,
        passages: Sequence[Any],
        filtered_triples: Optional[Sequence[Any]] = None,
    ) -> EvidenceSelectorResult:
        normalized_passages: List[Dict[str, Any]] = []

        for passage in passages:
            normalized = normalize_passage(passage)

            if normalized is not None:
                # Work on an isolated copy so a GRPO trajectory cannot mutate
                # evidence belonging to another trajectory.
                normalized_passages.append(deepcopy(normalized))

        triples = list(filtered_triples or [])

        effective_select_k = min(
            self.config.select_k,
            len(normalized_passages),
        )

        effective_top_pool = min(
            self.config.top_pool,
            len(normalized_passages),
        )

        if len(normalized_passages) <= self.config.select_k:
            return EvidenceSelectorResult(
                question=question,
                passages=normalized_passages,
                selected_indices=list(
                    range(1, len(normalized_passages) + 1)
                ),
                inferred_hops=[],
                passage_roles=[],
                chain_reason="",
                missing_evidence_warning="",
                skipped=True,
                fallback_used=False,
                error=None,
                raw_response=None,
                top_pool=effective_top_pool,
                select_k=effective_select_k,
            )

        prompt = build_prompt(
            question=question,
            passages=normalized_passages,
            filtered_triples=triples,
            top_pool=effective_top_pool,
            select_k=effective_select_k,
            max_passage_chars=self.config.max_passage_chars,
            max_triples=self.config.max_triples,
        )

        try:
            response_object, raw_response = self._call_llm(prompt)

            selected_indices = normalize_indices(
                response_object.get("selected_indices") or [],
                top_pool=effective_top_pool,
                select_k=effective_select_k,
            )

            selected_indices = fill_missing_indices(
                selected_indices=selected_indices,
                top_pool=effective_top_pool,
                select_k=effective_select_k,
            )

            reordered = reorder_by_selection(
                passages=normalized_passages,
                selected_indices=selected_indices,
                top_pool=effective_top_pool,
            )

            add_rank_metadata(
                original_passages=normalized_passages,
                reordered_passages=reordered,
                selected_indices=selected_indices,
                top_pool=effective_top_pool,
            )

            return EvidenceSelectorResult(
                question=question,
                passages=reordered,
                selected_indices=selected_indices,
                inferred_hops=list(
                    response_object.get("inferred_hops") or []
                ),
                passage_roles=list(
                    response_object.get("passage_roles") or []
                ),
                chain_reason=str(
                    response_object.get("chain_reason") or ""
                ),
                missing_evidence_warning=str(
                    response_object.get("missing_evidence_warning") or ""
                ),
                skipped=False,
                fallback_used=False,
                error=None,
                raw_response=raw_response,
                top_pool=effective_top_pool,
                select_k=effective_select_k,
            )

        except Exception as exc:
            if not self.config.fallback_to_original_order:
                raise

            return EvidenceSelectorResult(
                question=question,
                passages=normalized_passages,
                selected_indices=[],
                inferred_hops=[],
                passage_roles=[],
                chain_reason="",
                missing_evidence_warning="",
                skipped=False,
                fallback_used=True,
                error=str(exc),
                raw_response=None,
                top_pool=effective_top_pool,
                select_k=effective_select_k,
            )

    def select_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        new_row = deepcopy(row)

        passages_value = (
            row.get("evidence_passages")
            or row.get("retrieved_passages")
            or row.get("passages")
            or []
        )

        filtered_triples = list(row.get("filtered_triples") or [])

        if not filtered_triples:
            for step in row.get("steps") or []:
                if not isinstance(step, dict):
                    continue

                observation = step.get("observation") or {}
                if not isinstance(observation, dict):
                    continue

                search_result = observation.get("search_result") or {}
                if not isinstance(search_result, dict):
                    continue

                filtered_triples.extend(
                    search_result.get("filtered_triples") or []
                )

        question = str(
            row.get("question")
            or row.get("query")
            or ""
        )

        result = self.select(
            question=question,
            passages=passages_value,
            filtered_triples=filtered_triples,
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

        selector_metadata = result.metadata()
        selector_metadata["max_passage_chars"] = (
            self.config.max_passage_chars
        )
        selector_metadata["max_triples"] = self.config.max_triples

        if result.skipped:
            selector_metadata["reason"] = "num_passages <= select_k"

        if result.fallback_used:
            selector_metadata["fallback"] = "original_order"

        metadata["llm_evidence_selector_v2"] = selector_metadata

        if result.error is not None:
            metadata["llm_evidence_selector_v2_error"] = result.error

        return new_row
