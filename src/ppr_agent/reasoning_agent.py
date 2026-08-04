from __future__ import annotations
import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence
from .openie_extractor import OpenIEExtractorConfig, build_backend
from .schema import (
    AgentAction,
    AgentTrajectory,
    RetrievedPassage,
    SearchGraphRequest,
)

logger = logging.getLogger(__name__)
ReasoningBackend = Literal["vllm", "openai", "transformers", "mock"]

@dataclass
class ReasoningAgentConfig:
    backend: ReasoningBackend = "vllm"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Generation
    temperature: float = 0.0
    max_tokens: int = 512
    # Retrieval/action limits
    max_search_steps: int = 4
    default_top_k_triples: int = 50
    default_top_k_passages: int = 5
    # Prompt control
    max_evidence_passages_in_prompt: int = 8
    max_filtered_triples_in_prompt: int = 20
    max_candidate_triples_in_prompt: int = 10
    max_chars_per_passage: int = 900
    max_memory_text_chars: int = 1400
    # OpenAI / OpenAI-compatible endpoint
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    # vLLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True
    # Transformers
    device_map: str = "auto"


SYSTEM_PROMPT = """You are a multi-hop retrieval controller.
You control a graph retrieval tool called SearchGraph.
You can use exactly one of two actions:
1. SearchGraph
Use this when more evidence is needed.
You must provide:
- search_focus: the specific missing hop or missing fact
- seed_entities: entities to start graph search from
- relation_hints: relation clues that should guide graph search
2. SubmitFinalAnswer
Use this only when the evidence is sufficient to answer the question.
You must reason from BOTH:
- retrieved passages
- filtered triples
Important behavior:
- Do not repeat the same search if the previous evidence already gave a bridge entity.
- If evidence reveals a bridge entity, use that bridge entity as a seed in the next SearchGraph call.
- If evidence contains the full answer chain, stop and submit the answer.
- If evidence is incomplete, search for the missing hop only.
- Output only valid JSON.
- Do not include markdown.
- Do not include explanations outside JSON.
"""

USER_PROMPT_TEMPLATE = """Question:
{question}
Current step:
{step_id}
Maximum SearchGraph calls:
{max_search_steps}
Evidence collected so far:
{evidence_text}
Search/history summary:
{history_text}
Your task:
Decide the next action.

Evidence sufficiency checklist:
1. Identify what the question asks for.
2. Check whether the retrieved passages directly contain the answer.
3. Check whether the filtered triples reveal a bridge entity or missing relation.
4. If the answer chain is complete, submit the final answer.
5. If not complete, search for the specific missing hop using the best bridge entity as seed.

If evidence is incomplete, output JSON in this format:
{{
  "reasoning_summary": "short reason for what is missing",
  "action": "SearchGraph",
  "search_focus": "specific missing hop or evidence to search",
  "seed_entities": ["entity 1", "entity 2"],
  "relation_hints": ["relation clue 1", "relation clue 2"]
}}
If evidence is sufficient, output JSON in this format:
{{
  "reasoning_summary": "short reason why evidence is sufficient",
  "action": "SubmitFinalAnswer",
  "answer": "final answer"
}}
Rules:
- SearchGraph should be specific, not generic.
- Prefer bridge entities from the evidence, especially filtered triples.
- If this is step 0, start with main entities in the question.
- After step 0, do not keep using only the original question entities if a new bridge entity appears.
- If the same search was already tried and evidence did not change, choose a new seed entity or relation hint.
- If current step is near the limit, submit the best grounded answer if possible.
- If evidence truly does not contain the answer, submit "I don't know".
"""

@dataclass
class EvidenceView:
    passages: List[Any]
    filtered_triples: List[Any]
    candidate_triples: List[Any]
    seed_info: List[Any]


def _strip_code_fences(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:json|python)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


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


def parse_json_like_output(text: str) -> Dict[str, Any]:
    text = _strip_code_fences(text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    obj_text = _find_first_json_object(text)
    if obj_text is not None:
        try:
            parsed = json.loads(obj_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(obj_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


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
        if hasattr(first, "outputs"):
            try:
                return str(first.outputs[0].text)
            except Exception:
                return str(first)
        if isinstance(first, dict):
            for key in ["text", "output", "response", "content"]:
                if key in first:
                    return str(first[key])
        return str(first)

    if isinstance(raw, dict):
        for key in ["text", "output", "response", "content"]:
            if key in raw:
                return str(raw[key])
        return json.dumps(raw, ensure_ascii=False)

    if hasattr(raw, "outputs"):
        try:
            return str(raw.outputs[0].text)
        except Exception:
            return str(raw)

    return str(raw)


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    output: List[str] = []
    seen = set()

    for item in value:
        text = str(item).strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        output.append(text)
        seen.add(key)

    return output


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_plain(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items()}

    return str(obj)


def normalize_evidence_memory(evidence_memory: Any) -> EvidenceView:
    if evidence_memory is None:
        return EvidenceView(
            passages=[],
            filtered_triples=[],
            candidate_triples=[],
            seed_info=[],
        )

    if hasattr(evidence_memory, "passages"):
        passages = list(getattr(evidence_memory, "passages", []) or [])
        filtered_triples = list(getattr(evidence_memory, "filtered_triples", []) or [])
        candidate_triples = list(getattr(evidence_memory, "candidate_triples", []) or [])
        seed_info = list(getattr(evidence_memory, "seed_info", []) or [])

        return EvidenceView(
            passages=passages,
            filtered_triples=filtered_triples,
            candidate_triples=candidate_triples,
            seed_info=seed_info,
        )

    # Serialized dict
    if isinstance(evidence_memory, dict):
        return EvidenceView(
            passages=list(evidence_memory.get("passages", []) or []),
            filtered_triples=list(evidence_memory.get("filtered_triples", []) or []),
            candidate_triples=list(evidence_memory.get("candidate_triples", []) or []),
            seed_info=list(evidence_memory.get("seed_info", []) or []),
        )

    # Old list[RetrievedPassage] format
    if isinstance(evidence_memory, (list, tuple)):
        return EvidenceView(
            passages=list(evidence_memory),
            filtered_triples=[],
            candidate_triples=[],
            seed_info=[],
        )

    return EvidenceView(
        passages=[],
        filtered_triples=[],
        candidate_triples=[],
        seed_info=[],
    )


def format_passage_for_prompt(
    passage: Any,
    rank_fallback: int,
    max_chars_per_passage: int,
) -> str:
    passage_id = _safe_get(passage, "passage_id", _safe_get(passage, "id", "unknown"))
    title = _safe_get(passage, "title", None)
    rank = _safe_get(passage, "rank", rank_fallback)
    score = _safe_float(_safe_get(passage, "score", 0.0))
    text = _safe_get(passage, "text", _safe_get(passage, "passage", ""))

    title_line = f"Title: {title}\n" if title else ""

    return (
        f"[Passage rank={rank}, score={score:.6f}, id={passage_id}]\n"
        f"{title_line}"
        f"{_truncate_text(str(text), max_chars_per_passage)}"
    )


def format_triple_for_prompt(triple_obj: Any, rank: int, label: str) -> str:
    plain = _to_plain(triple_obj)

    score = None
    source = None
    triple = plain

    if isinstance(plain, dict):
        score = plain.get("score", plain.get("similarity", None))
        source = plain.get("source", plain.get("passage_id", None))
        triple = plain.get("triple", plain)

    if isinstance(triple, dict):
        subj = (
            triple.get("subject")
            or triple.get("subj")
            or triple.get("head")
            or triple.get("s")
            or ""
        )
        pred = (
            triple.get("predicate")
            or triple.get("relation")
            or triple.get("pred")
            or triple.get("p")
            or ""
        )
        obj = (
            triple.get("object")
            or triple.get("obj")
            or triple.get("tail")
            or triple.get("o")
            or ""
        )

        triple_text = f"({subj}, {pred}, {obj})"
    elif isinstance(triple, list) and len(triple) >= 3:
        triple_text = f"({triple[0]}, {triple[1]}, {triple[2]})"
    else:
        triple_text = str(triple)

    pieces = [f"[{label} rank={rank}"]

    if score is not None:
        pieces.append(f"score={score}")

    if source is not None:
        pieces.append(f"source={source}")

    header = ", ".join(pieces) + "]"

    return f"{header} {triple_text}"


def format_evidence_for_prompt(
    evidence_memory: Any,
    max_passages: int,
    max_filtered_triples: int,
    max_candidate_triples: int,
    max_chars_per_passage: int,
) -> str:
    view = normalize_evidence_memory(evidence_memory)

    if not view.passages and not view.filtered_triples and not view.candidate_triples:
        return "No evidence collected yet."

    sections: List[str] = []

    if view.filtered_triples:
        lines = ["FILTERED TRIPLES selected by the LLM triple filter:"]
        for i, triple in enumerate(view.filtered_triples[:max_filtered_triples], start=1):
            lines.append(format_triple_for_prompt(triple, rank=i, label="FilteredTriple"))
        sections.append("\n".join(lines))

    if view.passages:
        lines = ["RETRIEVED PASSAGES from PPR/SearchGraph:"]
        for i, passage in enumerate(view.passages[:max_passages], start=1):
            lines.append(
                format_passage_for_prompt(
                    passage=passage,
                    rank_fallback=i,
                    max_chars_per_passage=max_chars_per_passage,
                )
            )
        sections.append("\n\n".join(lines))

    if view.candidate_triples:
        lines = ["RAW CANDIDATE TRIPLES before filtering:"]
        for i, triple in enumerate(view.candidate_triples[:max_candidate_triples], start=1):
            lines.append(format_triple_for_prompt(triple, rank=i, label="CandidateTriple"))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def format_history_summary(evidence_memory: Any) -> str:
    view = normalize_evidence_memory(evidence_memory)

    passage_ids = []
    for p in view.passages:
        pid = _safe_get(p, "passage_id", _safe_get(p, "id", None))
        if pid is not None:
            passage_ids.append(str(pid))

    summary = {
        "num_passages": len(view.passages),
        "num_filtered_triples": len(view.filtered_triples),
        "num_candidate_triples": len(view.candidate_triples),
        "seen_passage_ids": passage_ids[:20],
    }

    return json.dumps(summary, ensure_ascii=False, indent=2)


def evidence_to_texts(
    evidence_memory: Any,
    max_chars: int = 1400,
) -> List[str]:
    view = normalize_evidence_memory(evidence_memory)

    texts: List[str] = []

    for p in view.passages:
        pid = _safe_get(p, "passage_id", _safe_get(p, "id", "unknown"))
        text = _safe_get(p, "text", _safe_get(p, "passage", ""))
        text = _truncate_text(str(text), max_chars)
        texts.append(f"[PASSAGE {pid}] {text}")

    for t in view.filtered_triples:
        texts.append(format_triple_for_prompt(t, rank=len(texts) + 1, label="FilteredTriple"))

    return texts


def collect_evidence_from_trajectory(
    trajectory: AgentTrajectory,
) -> List[RetrievedPassage]:
    evidence: List[RetrievedPassage] = []
    seen = set()

    for step in trajectory.steps:
        obs = step.observation
        if obs is None or obs.search_result is None:
            continue

        for passage in obs.search_result.passages:
            if passage.passage_id in seen:
                continue
            evidence.append(passage)
            seen.add(passage.passage_id)

    return evidence


def infer_seed_entities(question: str, evidence_memory: Any, max_entities: int = 6) -> List[str]:
    text_parts = [question]

    view = normalize_evidence_memory(evidence_memory)

    for triple in view.filtered_triples[:10]:
        text_parts.append(format_triple_for_prompt(triple, rank=0, label="Triple"))

    joined = "\n".join(text_parts)

    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,4}\b",
        joined,
    )

    bad = {
        "Question",
        "Evidence",
        "Passage",
        "FilteredTriple",
        "CandidateTriple",
        "SearchGraph",
        "SubmitFinalAnswer",
    }

    output: List[str] = []
    seen = set()

    for cand in candidates:
        cand = cand.strip(" .,:;()[]{}")
        if not cand or cand in bad:
            continue

        key = cand.lower()
        if key in seen:
            continue

        output.append(cand)
        seen.add(key)

        if len(output) >= max_entities:
            break

    return output


def normalize_action_name(action_name: Any) -> str:
    text = str(action_name or "").strip()

    mapping = {
        "searchgraph": "SearchGraph",
        "search_graph": "SearchGraph",
        "search graph": "SearchGraph",
        "submitfinalanswer": "SubmitFinalAnswer",
        "submit_final_answer": "SubmitFinalAnswer",
        "submit final answer": "SubmitFinalAnswer",
        "answer": "SubmitFinalAnswer",
        "stop": "SubmitFinalAnswer",
    }

    key = text.lower().replace("-", "_")
    return mapping.get(key, text)


class ReasoningAgent:
    def __init__(self, config: ReasoningAgentConfig):
        self.config = config

        if self.config.backend == "mock":
            self.backend = None
        else:
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


    def decide(
        self,
        question: str,
        step_id: int,
        evidence_memory: Optional[Any] = None,
        question_id: Optional[str] = None,
    ) -> AgentAction:

        if self.config.backend == "mock":
            return self._mock_decide(
                question=question,
                step_id=step_id,
                evidence_memory=evidence_memory,
            )

        messages = self.build_messages(
            question=question,
            step_id=step_id,
            evidence_memory=evidence_memory,
        )

        try:
            raw = self.backend.generate(
                messages=messages,
                max_tokens=self.config.max_tokens,
            )
            raw_output = normalize_generation_output(raw)
        except Exception as exc:
            logger.warning("Reasoning agent generation failed: %s", exc)
            raw_output = ""

        action = self.parse_action(
            raw_output=raw_output,
            question=question,
            step_id=step_id,
            evidence_memory=evidence_memory,
        )

        action.raw_output = raw_output
        return action

    def decide_from_trajectory(self, trajectory: AgentTrajectory) -> AgentAction:
        evidence_memory = collect_evidence_from_trajectory(trajectory)
        step_id = len(trajectory.steps)

        return self.decide(
            question=trajectory.question,
            question_id=trajectory.question_id,
            step_id=step_id,
            evidence_memory=evidence_memory,
        )

    def build_messages(
        self,
        question: str,
        step_id: int,
        evidence_memory: Optional[Any],
    ) -> List[Dict[str, str]]:
        evidence_text = format_evidence_for_prompt(
            evidence_memory=evidence_memory,
            max_passages=self.config.max_evidence_passages_in_prompt,
            max_filtered_triples=self.config.max_filtered_triples_in_prompt,
            max_candidate_triples=self.config.max_candidate_triples_in_prompt,
            max_chars_per_passage=self.config.max_chars_per_passage,
        )

        history_text = format_history_summary(evidence_memory)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            step_id=step_id,
            max_search_steps=self.config.max_search_steps,
            evidence_text=evidence_text,
            history_text=history_text,
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]


    def parse_action(
        self,
        raw_output: str,
        question: str,
        step_id: int,
        evidence_memory: Optional[Any] = None,
    ) -> AgentAction:
        parsed = parse_json_like_output(raw_output)
        action_name = normalize_action_name(parsed.get("action", ""))

        if action_name == "SubmitFinalAnswer":
            answer = str(parsed.get("answer", "")).strip()

            if answer:
                action = AgentAction(
                    action="SubmitFinalAnswer",
                    answer=answer,
                    reasoning_summary=str(parsed.get("reasoning_summary", "")).strip() or None,
                    raw_output=raw_output,
                )
                action.validate()
                return action

        if action_name == "SearchGraph":
            action = self._build_search_action_from_parsed(
                parsed=parsed,
                question=question,
                step_id=step_id,
                evidence_memory=evidence_memory,
                raw_output=raw_output,
            )
            action.validate()
            return action

        return self._fallback_action(
            question=question,
            step_id=step_id,
            evidence_memory=evidence_memory,
            raw_output=raw_output,
        )

    def _build_search_action_from_parsed(
        self,
        parsed: Dict[str, Any],
        question: str,
        step_id: int,
        evidence_memory: Optional[Any],
        raw_output: Optional[str],
    ) -> AgentAction:
        seed_entities = _as_string_list(parsed.get("seed_entities", []))
        relation_hints = _as_string_list(parsed.get("relation_hints", []))
        search_focus = str(parsed.get("search_focus", "")).strip()

        if not search_focus:
            search_focus = question

        # If LLM forgets seeds, infer them from question + filtered triples.
        if not seed_entities:
            seed_entities = infer_seed_entities(
                question=question,
                evidence_memory=evidence_memory,
                max_entities=6,
            )

        evidence_texts = evidence_to_texts(
            evidence_memory=evidence_memory,
            max_chars=self.config.max_memory_text_chars,
        )

        search_request = SearchGraphRequest(
            question=question,
            search_focus=search_focus,
            seed_entities=seed_entities,
            relation_hints=relation_hints,
            evidence_so_far=evidence_texts,
            top_k_triples=self.config.default_top_k_triples,
            top_k_passages=self.config.default_top_k_passages,
            step_id=step_id,
        )

        return AgentAction(
            action="SearchGraph",
            search_request=search_request,
            reasoning_summary=str(parsed.get("reasoning_summary", "")).strip() or None,
            raw_output=raw_output,
        )


    def _fallback_action(
        self,
        question: str,
        step_id: int,
        evidence_memory: Optional[Any],
        raw_output: Optional[str],
    ) -> AgentAction:
        if step_id < self.config.max_search_steps:
            seed_entities = infer_seed_entities(
                question=question,
                evidence_memory=evidence_memory,
                max_entities=6,
            )

            search_request = SearchGraphRequest(
                question=question,
                search_focus=question,
                seed_entities=seed_entities,
                relation_hints=[],
                evidence_so_far=evidence_to_texts(
                    evidence_memory=evidence_memory,
                    max_chars=self.config.max_memory_text_chars,
                ),
                top_k_triples=self.config.default_top_k_triples,
                top_k_passages=self.config.default_top_k_passages,
                step_id=step_id,
            )

            return AgentAction(
                action="SearchGraph",
                search_request=search_request,
                reasoning_summary="Fallback: model output was not valid JSON, so search the graph.",
                raw_output=raw_output,
            )

        return AgentAction(
            action="SubmitFinalAnswer",
            answer="I don't know",
            reasoning_summary="Fallback: search budget exhausted and no valid final answer was produced.",
            raw_output=raw_output,
        )

    def _mock_decide(
        self,
        question: str,
        step_id: int,
        evidence_memory: Optional[Any],
    ) -> AgentAction:
       
        if step_id == 0:
            search_request = SearchGraphRequest(
                question=question,
                search_focus=question,
                seed_entities=infer_seed_entities(question, evidence_memory, max_entities=6),
                relation_hints=[],
                evidence_so_far=[],
                top_k_triples=self.config.default_top_k_triples,
                top_k_passages=self.config.default_top_k_passages,
                step_id=step_id,
            )

            return AgentAction(
                action="SearchGraph",
                search_request=search_request,
                reasoning_summary="Mock policy: run initial graph search.",
                raw_output=None,
            )

        return AgentAction(
            action="SubmitFinalAnswer",
            answer="I don't know",
            reasoning_summary="Mock policy: stop after one search.",
            raw_output=None,
        )


def decide_next_action(
    question: str,
    evidence_memory: Optional[Any] = None,
    step_id: int = 0,
    backend: ReasoningBackend = "mock",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> AgentAction:
    agent = ReasoningAgent(
        ReasoningAgentConfig(
            backend=backend,
            model_name=model_name,
        )
    )

    return agent.decide(
        question=question,
        step_id=step_id,
        evidence_memory=evidence_memory,
    )