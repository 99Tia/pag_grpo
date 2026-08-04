from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ppr_agent.openie_extractor import OpenIEExtractorConfig, build_backend


@dataclass
class AnswerReaderConfig:

    backend: str = "openai"
    model_name: str = "llama70b-filter"

    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"

    temperature: float = 0.0
    max_output_tokens: int = 256

    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.65
    trust_remote_code: bool = False
    device_map: str = "auto"

    top_k_evidence: int = 5
    top_k_filtered_triples: int = 20
    max_passage_chars: int = 2500

    validate_support_ids: bool = True

    def validate(self) -> None:
        if self.backend not in {
            "openai",
            "vllm",
            "transformers",
            "mock",
        }:
            raise ValueError(
                "backend must be one of: "
                "openai, vllm, transformers, mock."
            )

        if not self.model_name:
            raise ValueError("model_name must be provided.")

        if self.max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be greater than zero."
            )

        if self.top_k_evidence <= 0:
            raise ValueError(
                "top_k_evidence must be greater than zero."
            )

        if self.top_k_filtered_triples < 0:
            raise ValueError(
                "top_k_filtered_triples cannot be negative."
            )

        if self.max_passage_chars <= 0:
            raise ValueError(
                "max_passage_chars must be greater than zero."
            )


@dataclass
class AnswerReaderResult:
    question: str
    predicted_answer: str

    supporting_passage_ids: List[str]
    invalid_supporting_passage_ids: List[str]
    supporting_triples: List[str]

    confidence: Optional[float]

    evidence_passages: List[Dict[str, Any]]
    filtered_triples: List[Any]

    raw_response: str
    parsed_response: Dict[str, Any]
    parse_success: bool

    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "predicted_answer": self.predicted_answer,
            "supporting_passage_ids": list(
                self.supporting_passage_ids
            ),
            "invalid_supporting_passage_ids": list(
                self.invalid_supporting_passage_ids
            ),
            "supporting_triples": list(
                self.supporting_triples
            ),
            "confidence": self.confidence,
            "evidence_passages": self.evidence_passages,
            "filtered_triples": self.filtered_triples,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "parse_success": self.parse_success,
            "metadata": dict(self.metadata),
        }


def to_plain(value: Any) -> Any:
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


def safe_get(
    row: Dict[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]

    return default


def as_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def truncate_text(text: Any, max_chars: int) -> str:
    output = str(text or "").strip()

    if max_chars > 0 and len(output) > max_chars:
        return output[:max_chars].rstrip() + " ..."

    return output


def normalize_generation_output(raw: Any) -> str:
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw

    if isinstance(raw, list):
        if not raw:
            return ""

        first = raw[0]

        if isinstance(first, str):
            return first

        if isinstance(first, dict):
            for key in [
                "text",
                "output",
                "response",
                "content",
            ]:
                if key in first:
                    return str(first[key])

            return json.dumps(
                first,
                ensure_ascii=False,
            )

        if hasattr(first, "outputs"):
            try:
                return str(first.outputs[0].text)
            except Exception:
                return str(first)

        return str(first)

    if isinstance(raw, dict):
        for key in [
            "text",
            "output",
            "response",
            "content",
        ]:
            if key in raw:
                return str(raw[key])

        return json.dumps(
            raw,
            ensure_ascii=False,
        )

    if hasattr(raw, "outputs"):
        try:
            return str(raw.outputs[0].text)
        except Exception:
            return str(raw)

    return str(raw)


def normalize_passage(
    raw: Any,
    fallback_rank: int,
) -> Optional[Dict[str, Any]]:
    plain = to_plain(raw)

    if plain is None:
        return None

    if isinstance(plain, str):
        text = plain.strip()

        if not text:
            return None

        return {
            "passage_id": f"unknown-{fallback_rank}",
            "title": None,
            "text": text,
            "score": None,
            "rank": fallback_rank,
            "metadata": {},
        }

    if not isinstance(plain, dict):
        return None

    text = safe_get(
        plain,
        [
            "text",
            "passage",
            "passage_text",
            "paragraph_text",
            "content",
            "body",
            "paragraph",
        ],
        "",
    )

    text = str(text).strip()

    if not text:
        return None

    metadata = plain.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    passage_id = safe_get(
        plain,
        [
            "passage_id",
            "id",
            "idx",
            "chunk_id",
            "node_id",
            "fallback_idx",
        ],
        None,
    )

    if passage_id is None:
        passage_id = safe_get(
            metadata,
            [
                "passage_id",
                "idx",
                "passage_idx",
                "corpus_idx",
                "source_idx",
                "node_id",
                "fallback_idx",
            ],
            f"unknown-{fallback_rank}",
        )

    title = safe_get(
        plain,
        ["title", "name"],
        None,
    )

    if title is None:
        title = metadata.get("title")

    score = safe_get(
        plain,
        [
            "score",
            "ppr_score",
            "retrieval_score",
        ],
        None,
    )

    if score is None:
        score = safe_get(
            metadata,
            [
                "score",
                "ppr_score",
                "retrieval_score",
            ],
            None,
        )

    rank = safe_get(
        plain,
        ["rank"],
        None,
    )

    if rank is None:
        rank = safe_get(
            metadata,
            [
                "rank",
                "llm_selector_v2_new_rank",
                "hybrid_ppr_llmselect_final_rank",
            ],
            fallback_rank,
        )

    return {
        "passage_id": str(passage_id),
        "title": title,
        "text": text,
        "score": score,
        "rank": rank,
        "metadata": metadata,
    }


def collect_evidence_passages(
    passages: Sequence[Any],
    top_k: int,
    max_passage_chars: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for index, item in enumerate(passages, start=1):
        passage = normalize_passage(
            item,
            fallback_rank=index,
        )

        if passage is None:
            continue

        passage_id = passage["passage_id"]

        if passage_id in seen:
            continue

        seen.add(passage_id)

        passage["text"] = truncate_text(
            passage["text"],
            max_passage_chars,
        )

        output.append(passage)

        if len(output) >= top_k:
            break

    return output


def collect_filtered_triples(
    triples: Sequence[Any],
    top_k: int,
) -> List[Any]:
    if top_k <= 0:
        return []

    output: List[Any] = []
    seen: Set[str] = set()

    for triple in triples:
        plain = to_plain(triple)

        key = json.dumps(
            plain,
            ensure_ascii=False,
            sort_keys=True,
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(plain)

        if len(output) >= top_k:
            break

    return output


def format_triple_for_prompt(
    triple_obj: Any,
    index: int,
) -> str:
    plain = to_plain(triple_obj)
    score = None
    triple = plain

    if isinstance(plain, dict):
        score = (
            plain.get("original_score")
            or plain.get("score")
        )

        triple = plain.get(
            "triple",
            plain,
        )

    if isinstance(triple, dict):
        subject = (
            triple.get("subject")
            or triple.get("subj")
            or triple.get("head")
            or ""
        )

        predicate = (
            triple.get("predicate")
            or triple.get("relation")
            or triple.get("pred")
            or ""
        )

        object_ = (
            triple.get("object")
            or triple.get("obj")
            or triple.get("tail")
            or ""
        )

        text = (
            f"({subject}, {predicate}, {object_})"
        )

    elif (
        isinstance(triple, list)
        and len(triple) >= 3
    ):
        text = (
            f"({triple[0]}, "
            f"{triple[1]}, "
            f"{triple[2]})"
        )

    else:
        text = str(triple)

    if score is not None:
        return (
            f"[T{index} score={score}] {text}"
        )

    return f"[T{index}] {text}"


def build_answer_messages(
    question: str,
    evidence_passages: Sequence[Dict[str, Any]],
    filtered_triples: Sequence[Any],
) -> List[Dict[str, str]]:
    triple_lines: List[str] = []

    for index, triple in enumerate(
        filtered_triples,
        start=1,
    ):
        triple_lines.append(
            format_triple_for_prompt(
                triple,
                index,
            )
        )

    triples_text = (
        "\n".join(triple_lines)
        if triple_lines
        else "No filtered triples were saved."
    )

    passage_blocks: List[str] = []

    for index, passage in enumerate(
        evidence_passages,
        start=1,
    ):
        passage_id = passage.get(
            "passage_id",
            f"passage-{index}",
        )

        title = passage.get("title")
        text = passage.get("text", "")

        header = (
            f"[P{index}] passage_id={passage_id}"
        )

        if title:
            header += f" | title={title}"

        passage_blocks.append(
            f"{header}\n{text}"
        )

    passages_text = (
        "\n\n".join(passage_blocks)
        if passage_blocks
        else "No evidence passages were retrieved."
    )

    system_prompt = """You are a precise multi-hop question-answering model.

You must answer using ONLY the provided filtered triples and evidence passages.
Do not invent facts.
If the evidence does not contain the answer, answer exactly: I don't know.
Return only valid JSON.
"""

    user_prompt = f"""Question:
{question}

Filtered triples:
{triples_text}

Evidence passages:
{passages_text}

Return ONLY valid JSON with this schema:
{{
  "answer": "short final answer string",
  "supporting_passage_ids": ["passage_id"],
  "supporting_triples": ["triple ids or short triple text"],
  "confidence": 0.0
}}

Rules:
- The answer should be the shortest correct answer phrase.
- Use the passages and triples together.
- If one passage identifies a bridge entity and another passage gives the requested fact about that entity, combine them.
- supporting_passage_ids must come from the passage_id values shown above.
- Do not include explanation outside JSON.
"""

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def strip_code_fences(text: str) -> str:
    output = str(text or "").strip()

    output = re.sub(
        r"^```(?:json)?",
        "",
        output,
        flags=re.IGNORECASE,
    ).strip()

    output = re.sub(
        r"```$",
        "",
        output,
    ).strip()

    return output


def extract_json_object(
    text: str,
) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from the answer reader response."""

    cleaned = strip_code_fences(text)

    try:
        value = json.loads(cleaned)

        if isinstance(value, dict):
            return value

    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]

        try:
            value = json.loads(candidate)

            if isinstance(value, dict):
                return value

        except Exception:
            return None

    return None


def parse_answer_response(
    raw_response: str,
) -> Tuple[
    str,
    List[str],
    List[str],
    Optional[float],
    Dict[str, Any],
    bool,
]:
    parsed = extract_json_object(raw_response)

    if parsed is None:
        cleaned = strip_code_fences(
            raw_response
        ).strip()

        first_line = (
            cleaned.split("\n")[0].strip()
            if cleaned
            else ""
        )

        return (
            first_line,
            [],
            [],
            None,
            {},
            False,
        )

    answer = (
        parsed.get("answer")
        or parsed.get("final_answer")
        or parsed.get("predicted_answer")
        or ""
    )

    supporting_passage_ids = parsed.get(
        "supporting_passage_ids"
    )

    if not isinstance(
        supporting_passage_ids,
        list,
    ):
        supporting_passage_ids = []

    supporting_triples = parsed.get(
        "supporting_triples"
    )

    if not isinstance(
        supporting_triples,
        list,
    ):
        supporting_triples = []

    confidence = parsed.get("confidence")

    if isinstance(confidence, (int, float)):
        confidence_value: Optional[float] = float(
            confidence
        )
        confidence_value = max(
            0.0,
            min(1.0, confidence_value),
        )
    else:
        confidence_value = None

    return (
        str(answer).strip(),
        [
            str(item)
            for item in supporting_passage_ids
        ],
        [
            str(item)
            for item in supporting_triples
        ],
        confidence_value,
        parsed,
        True,
    )


def validate_supporting_passage_ids(
    supporting_passage_ids: Sequence[str],
    evidence_passages: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:

    valid_ids = {
        str(passage.get("passage_id"))
        for passage in evidence_passages
        if passage.get("passage_id") is not None
    }

    valid: List[str] = []
    invalid: List[str] = []
    seen_valid: Set[str] = set()
    seen_invalid: Set[str] = set()

    for value in supporting_passage_ids:
        passage_id = str(value)

        if passage_id in valid_ids:
            if passage_id not in seen_valid:
                valid.append(passage_id)
                seen_valid.add(passage_id)
        else:
            if passage_id not in seen_invalid:
                invalid.append(passage_id)
                seen_invalid.add(passage_id)

    return valid, invalid


class MockAnswerBackend:
    """Deterministic backend used for integration tests."""

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> str:
        del messages, max_tokens, kwargs

        return json.dumps(
            {
                "answer": "I don't know",
                "supporting_passage_ids": [],
                "supporting_triples": [],
                "confidence": 0.0,
            }
        )


def build_answer_backend(
    config: AnswerReaderConfig,
) -> Any:
    """Build the same backend family used by the original script."""

    if config.backend == "mock":
        return MockAnswerBackend()

    backend_config = OpenIEExtractorConfig(
        backend=config.backend,
        model_name=config.model_name,
        temperature=config.temperature,
        max_ner_tokens=config.max_output_tokens,
        max_triple_tokens=config.max_output_tokens,
        api_key_env=config.api_key_env,
        base_url=config.base_url,
        tensor_parallel_size=(
            config.tensor_parallel_size
        ),
        gpu_memory_utilization=(
            config.gpu_memory_utilization
        ),
        trust_remote_code=(
            config.trust_remote_code
        ),
        device_map=config.device_map,
    )

    return build_backend(backend_config)


def generate_one_answer(
    backend: Any,
    messages: Sequence[Dict[str, str]],
    max_output_tokens: int,
) -> str:
    try:
        raw = backend.generate(
            messages=list(messages),
            max_tokens=max_output_tokens,
        )

        return normalize_generation_output(raw)

    except TypeError:
        prompt = "\n\n".join(
            f"{message['role'].upper()}:\n"
            f"{message['content']}"
            for message in messages
        )

        raw = backend.generate(
            prompt,
            max_tokens=max_output_tokens,
        )

        return normalize_generation_output(raw)


class GroundedAnswerReader:

    def __init__(
        self,
        config: AnswerReaderConfig,
        backend: Optional[Any] = None,
    ):
        config.validate()

        self.config = config
        self.backend = (
            backend
            if backend is not None
            else build_answer_backend(config)
        )

    def generate(
        self,
        question: str,
        passages: Sequence[Any],
        filtered_triples: Optional[
            Sequence[Any]
        ] = None,
    ) -> AnswerReaderResult:
        evidence_passages = (
            collect_evidence_passages(
                passages=passages,
                top_k=self.config.top_k_evidence,
                max_passage_chars=(
                    self.config.max_passage_chars
                ),
            )
        )

        selected_triples = collect_filtered_triples(
            triples=list(filtered_triples or []),
            top_k=(
                self.config.top_k_filtered_triples
            ),
        )

        messages = build_answer_messages(
            question=question,
            evidence_passages=evidence_passages,
            filtered_triples=selected_triples,
        )

        raw_response = generate_one_answer(
            backend=self.backend,
            messages=messages,
            max_output_tokens=(
                self.config.max_output_tokens
            ),
        )

        (
            predicted_answer,
            requested_support_ids,
            supporting_triples,
            confidence,
            parsed_response,
            parse_success,
        ) = parse_answer_response(raw_response)

        if self.config.validate_support_ids:
            (
                supporting_passage_ids,
                invalid_supporting_passage_ids,
            ) = validate_supporting_passage_ids(
                supporting_passage_ids=(
                    requested_support_ids
                ),
                evidence_passages=evidence_passages,
            )
        else:
            supporting_passage_ids = list(
                requested_support_ids
            )
            invalid_supporting_passage_ids = []

        metadata = {
            "answer_model": self.config.model_name,
            "answer_backend": self.config.backend,
            "top_k_evidence": (
                self.config.top_k_evidence
            ),
            "top_k_filtered_triples": (
                self.config.top_k_filtered_triples
            ),
            "max_passage_chars": (
                self.config.max_passage_chars
            ),
            "parse_success": parse_success,
            "num_evidence_passages": len(
                evidence_passages
            ),
            "num_filtered_triples": len(
                selected_triples
            ),
            "num_invalid_support_ids": len(
                invalid_supporting_passage_ids
            ),
        }

        return AnswerReaderResult(
            question=question,
            predicted_answer=predicted_answer,
            supporting_passage_ids=(
                supporting_passage_ids
            ),
            invalid_supporting_passage_ids=(
                invalid_supporting_passage_ids
            ),
            supporting_triples=supporting_triples,
            confidence=confidence,
            evidence_passages=evidence_passages,
            filtered_triples=selected_triples,
            raw_response=raw_response,
            parsed_response=parsed_response,
            parse_success=parse_success,
            metadata=metadata,
        )

    def generate_from_row(
        self,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:

        source_row = row
        trajectory = row.get("trajectory")

        if isinstance(trajectory, dict):
            working_row = dict(trajectory)

            for key in [
                "question_id",
                "question",
                "gold_answers",
                "gold_passage_ids",
                "controller_final_answer",
                "answer_for_reward",
                "answer_source",
            ]:
                if (
                    key in source_row
                    and key not in working_row
                ):
                    working_row[key] = source_row[key]
        else:
            working_row = row

        question = str(
            safe_get(
                working_row,
                ["question", "query"],
                "",
            )
        )

        passages = (
            working_row.get("evidence_passages")
            or working_row.get(
                "retrieved_passages"
            )
            or working_row.get("passages")
            or []
        )

        filtered_triples = list(
            working_row.get("filtered_triples")
            or []
        )

        if not filtered_triples:
            for step in (
                working_row.get("steps")
                or []
            ):
                if not isinstance(step, dict):
                    continue

                observation = (
                    step.get("observation")
                    or {}
                )

                if not isinstance(
                    observation,
                    dict,
                ):
                    continue

                search_result = (
                    observation.get(
                        "search_result"
                    )
                    or {}
                )

                if not isinstance(
                    search_result,
                    dict,
                ):
                    continue

                filtered_triples.extend(
                    search_result.get(
                        "filtered_triples"
                    )
                    or []
                )

        result = self.generate(
            question=question,
            passages=passages,
            filtered_triples=filtered_triples,
        )

        question_id = str(
            safe_get(
                working_row,
                ["question_id", "id", "qid"],
                "",
            )
        )

        return {
            "question_id": question_id,
            "question": question,
            "predicted_answer": (
                result.predicted_answer
            ),
            "gold_answers": as_string_list(
                safe_get(
                    working_row,
                    [
                        "gold_answers",
                        "answers",
                        "answer",
                    ],
                    [],
                )
            ),
            "gold_passage_ids": as_string_list(
                safe_get(
                    working_row,
                    [
                        "gold_passage_ids",
                        "supporting_passage_ids",
                    ],
                    [],
                )
            ),
            "controller_final_answer": (
                working_row.get(
                    "controller_final_answer"
                )
                or working_row.get(
                    "final_answer"
                )
                or source_row.get(
                    "controller_final_answer"
                )
            ),
            "supporting_passage_ids": (
                result.supporting_passage_ids
            ),
            "invalid_supporting_passage_ids": (
                result.invalid_supporting_passage_ids
            ),
            "supporting_triples": (
                result.supporting_triples
            ),
            "confidence": result.confidence,
            "evidence_passages": (
                result.evidence_passages
            ),
            "filtered_triples": (
                result.filtered_triples
            ),
            "raw_response": result.raw_response,
            "parsed_response": (
                result.parsed_response
            ),
            "parse_success": (
                result.parse_success
            ),
            "metadata": {
                **result.metadata,
                "source_format": (
                    "grpo_rollout"
                    if isinstance(
                        source_row.get("trajectory"),
                        dict,
                    )
                    else "retrieval_trajectory"
                ),
            },
        }
