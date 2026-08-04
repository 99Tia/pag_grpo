from __future__ import annotations
import ast
import difflib
import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from .openie_extractor import OpenIEExtractorConfig, build_backend, normalize_triples
from .schema import (
    CandidateTriple,
    FilteredTriple,
    Triple,
    TripleFilterResult,
    normalize_text,
)

logger = logging.getLogger(__name__)

TripleFilterBackend = Literal["vllm", "openai", "transformers", "mock"]


@dataclass
class TripleFilterConfig:
    backend: TripleFilterBackend = "vllm"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"

    dspy_prompt_path: Optional[str] = None

    # Filtering behavior
    enabled: bool = True
    max_candidates_in_prompt: int = 50
    max_output_triples: Optional[int] = None
    fallback_to_input_if_empty: bool = False

    # Generation
    temperature: float = 0.0
    max_tokens: int = 512

    # OpenAI / OpenAI-compatible endpoint
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None

    # vLLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True

    # Transformers
    device_map: str = "auto"


DEFAULT_SYSTEM_PROMPT = """You are a careful fact filtering module for multi-hop retrieval.

Your job is to select only the facts that are useful for answering the question.
A fact is useful if it helps identify an answer, a bridge entity, a needed relation, or a supporting passage.

Return only the selected facts.
Do not explain.
Do not invent new facts.
Use only facts from the candidate list.
"""

ONE_INPUT_TEMPLATE = """[[ ## question ## ]]
{question}

[[ ## fact_before_filter ## ]]
{fact_before_filter}

Respond with the corresponding output field, starting with `[[ ## fact_after_filter ## ]]`.

The output must be a valid Python/JSON object in this exact format:
{{"fact": [["subject", "predicate", "object"], ...]}}

Then end with:
[[ ## completed ## ]]
"""

ONE_OUTPUT_TEMPLATE = """[[ ## fact_after_filter ## ]]
{fact_after_filter}

[[ ## completed ## ]]"""


FIELD_HEADER_PATTERN = re.compile(r"\[\[ ## (\w+) ## \]\]")

def _strip_code_fences(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:json|python)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text

def _extract_dspy_field(response: str, field_name: str) -> Optional[str]:
    sections: List[Tuple[Optional[str], List[str]]] = [(None, [])]

    for line in str(response).splitlines():
        match = FIELD_HEADER_PATTERN.match(line.strip())
        if match:
            sections.append((match.group(1), []))
        else:
            sections[-1][1].append(line)

    for key, lines in sections:
        if key == field_name:
            return "\n".join(lines).strip()

    return None


def _find_first_json_object(text: str) -> Optional[str]:
    text = _strip_code_fences(text)

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _parse_object_or_list(text: str) -> Any:
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    # Then first JSON object inside text.
    obj = _find_first_json_object(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except Exception:
            try:
                return ast.literal_eval(obj)
            except Exception:
                pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return None


def parse_filtered_facts(response: str) -> List[Tuple[str, str, str]]:
    field_value = _extract_dspy_field(response, "fact_after_filter")

    if field_value is None:
        field_value = response

    parsed = _parse_object_or_list(field_value)

    if isinstance(parsed, dict):
        raw_facts = parsed.get("fact", [])
    elif isinstance(parsed, list):
        raw_facts = parsed
    else:
        raw_facts = []

    return normalize_triples(raw_facts)


def _candidate_tuple(candidate: CandidateTriple) -> Tuple[str, str, str]:
    return (
        candidate.triple.subject,
        candidate.triple.predicate,
        candidate.triple.object,
    )


def _normalized_tuple(triple_like: Tuple[str, str, str]) -> Tuple[str, str, str]:
    return tuple(normalize_text(x) for x in triple_like)  # type: ignore[return-value]


def _candidate_key(candidate: CandidateTriple) -> Tuple[str, str, str]:
    return _normalized_tuple(_candidate_tuple(candidate))


def _format_fact_before_filter(candidates: Sequence[CandidateTriple]) -> str:
    payload = {
        "fact": [
            [
                candidate.triple.subject,
                candidate.triple.predicate,
                candidate.triple.object,
            ]
            for candidate in candidates
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


class TripleFilterPromptBuilder:

    def __init__(self, dspy_prompt_path: Optional[str] = None):
        self.dspy_prompt_path = dspy_prompt_path
        self.message_template = self._make_template(dspy_prompt_path)

    def _make_template(self, dspy_prompt_path: Optional[str]) -> List[Dict[str, str]]:
        if dspy_prompt_path is None:
            return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        if not os.path.exists(dspy_prompt_path):
            logger.warning("DSPy prompt file not found: %s", dspy_prompt_path)
            return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        try:
            with open(dspy_prompt_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            program = payload.get("prog", {})
            system_prompt = program.get("system", DEFAULT_SYSTEM_PROMPT)
            demos = program.get("demos", [])

            messages = [{"role": "system", "content": system_prompt}]

            for demo in demos:
                question = demo.get("question", "")
                fact_before_filter = demo.get("fact_before_filter", "")
                fact_after_filter = demo.get("fact_after_filter", '{"fact": []}')

                messages.append(
                    {
                        "role": "user",
                        "content": ONE_INPUT_TEMPLATE.format(
                            question=question,
                            fact_before_filter=fact_before_filter,
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": ONE_OUTPUT_TEMPLATE.format(
                            fact_after_filter=fact_after_filter,
                        ),
                    }
                )

            return messages

        except Exception as exc:
            logger.warning("Could not load DSPy prompt file %s: %s", dspy_prompt_path, exc)
            return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    def build_messages(
        self,
        question: str,
        candidates: Sequence[CandidateTriple],
    ) -> List[Dict[str, str]]:
        messages = deepcopy(self.message_template)
        messages.append(
            {
                "role": "user",
                "content": ONE_INPUT_TEMPLATE.format(
                    question=question,
                    fact_before_filter=_format_fact_before_filter(candidates),
                ),
            }
        )
        return messages


class TripleFilter:
    def __init__(self, config: TripleFilterConfig):
        self.config = config
        self.prompt_builder = TripleFilterPromptBuilder(config.dspy_prompt_path)

        openie_like_config = OpenIEExtractorConfig(
            backend=config.backend,
            model_name=config.model_name,
            temperature=config.temperature,
            max_ner_tokens=config.max_tokens,
            max_triple_tokens=config.max_tokens,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map,
        )

        self.backend = build_backend(openie_like_config)

    def __call__(
        self,
        question: str,
        candidates: Sequence[CandidateTriple],
        len_after_filter: Optional[int] = None,
    ) -> TripleFilterResult:
        return self.filter(
            question=question,
            candidates=candidates,
            len_after_filter=len_after_filter,
        )

    def filter(
        self,
        question: str,
        candidates: Sequence[CandidateTriple],
        len_after_filter: Optional[int] = None,
    ) -> TripleFilterResult:
        candidates = list(candidates)

        if len_after_filter is None:
            len_after_filter = self.config.max_output_triples

        if self.config.max_candidates_in_prompt > 0:
            prompt_candidates = candidates[: self.config.max_candidates_in_prompt]
        else:
            prompt_candidates = candidates

        if not candidates:
            return TripleFilterResult(
                query=question,
                candidate_triples=[],
                filtered_triples=[],
                raw_response=None,
                metadata={"reason": "no_candidates"},
            )

        if not self.config.enabled:
            kept = self._keep_input_candidates(
                prompt_candidates,
                len_after_filter=len_after_filter,
            )
            return TripleFilterResult(
                query=question,
                candidate_triples=prompt_candidates,
                filtered_triples=kept,
                raw_response=None,
                metadata={"filter_enabled": False},
            )

        messages = self.prompt_builder.build_messages(
            question=question,
            candidates=prompt_candidates,
        )

        try:
            raw_response = self.backend.generate(
                messages=messages,
                max_tokens=self.config.max_tokens,
            )
            generated_facts = parse_filtered_facts(raw_response)

        except Exception as exc:
            logger.warning("Triple filter failed: %s", exc)
            raw_response = None
            generated_facts = []

        filtered = self._match_generated_facts_to_candidates(
            generated_facts=generated_facts,
            candidates=prompt_candidates,
        )

        if not filtered and self.config.fallback_to_input_if_empty:
            filtered = self._keep_input_candidates(
                prompt_candidates,
                len_after_filter=len_after_filter,
            )

        if len_after_filter is not None:
            filtered = filtered[:len_after_filter]

        return TripleFilterResult(
            query=question,
            candidate_triples=prompt_candidates,
            filtered_triples=filtered,
            raw_response=raw_response,
            metadata={
                "filter_enabled": True,
                "num_candidates": len(prompt_candidates),
                "num_generated_facts": len(generated_facts),
                "num_filtered": len(filtered),
                "fallback_to_input_if_empty": self.config.fallback_to_input_if_empty,
            },
        )

    def _keep_input_candidates(
        self,
        candidates: Sequence[CandidateTriple],
        len_after_filter: Optional[int] = None,
    ) -> List[FilteredTriple]:
        kept: List[FilteredTriple] = []

        if len_after_filter is not None:
            candidates = candidates[:len_after_filter]

        for rank, candidate in enumerate(candidates, start=1):
            kept.append(
                FilteredTriple(
                    triple=candidate.triple,
                    original_index=candidate.index,
                    original_score=candidate.score,
                    filter_rank=rank,
                    filter_confidence=None,
                )
            )

        return kept

    def _match_generated_facts_to_candidates(
        self,
        generated_facts: Sequence[Tuple[str, str, str]],
        candidates: Sequence[CandidateTriple],
    ) -> List[FilteredTriple]:
        if not generated_facts:
            return []

        candidate_keys = [_candidate_key(candidate) for candidate in candidates]
        key_to_position: Dict[Tuple[str, str, str], int] = {
            key: i for i, key in enumerate(candidate_keys)
        }

        candidate_strings = [str(key) for key in candidate_keys]

        selected_positions: List[int] = []
        selected_seen = set()

        for generated in generated_facts:
            gen_key = _normalized_tuple(generated)

            position = key_to_position.get(gen_key)

            if position is None and candidate_strings:
                closest = difflib.get_close_matches(
                    str(gen_key),
                    candidate_strings,
                    n=1,
                    cutoff=0.0,
                )
                if closest:
                    position = candidate_strings.index(closest[0])

            if position is None:
                continue

            if position in selected_seen:
                continue

            selected_positions.append(position)
            selected_seen.add(position)

        filtered: List[FilteredTriple] = []

        for rank, position in enumerate(selected_positions, start=1):
            candidate = candidates[position]
            filtered.append(
                FilteredTriple(
                    triple=candidate.triple,
                    original_index=candidate.index,
                    original_score=candidate.score,
                    filter_rank=rank,
                    filter_confidence=None,
                )
            )

        return filtered


def filter_candidate_triples(
    question: str,
    candidates: Sequence[CandidateTriple],
    backend: TripleFilterBackend = "vllm",
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    dspy_prompt_path: Optional[str] = None,
    max_output_triples: Optional[int] = None,
) -> TripleFilterResult:
    config = TripleFilterConfig(
        backend=backend,
        model_name=model_name,
        dspy_prompt_path=dspy_prompt_path,
        max_output_triples=max_output_triples,
    )
    triple_filter = TripleFilter(config)
    return triple_filter.filter(question=question, candidates=candidates)
