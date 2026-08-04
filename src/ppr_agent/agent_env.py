from __future__ import annotations
import json
import logging
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
from .ppr_search import PPRSearchEngine
from .reasoning_agent import ReasoningAgent
from .schema import (
    AgentAction,
    AgentObservation,
    AgentStep,
    AgentTrajectory,
    RetrievedPassage,
    SearchGraphRequest,
)
from .answer_reader import GroundedAnswerReader
from .evidence_fusion import HybridEvidenceFuser
from .evidence_selector import EvidenceSelectorV2

logger = logging.getLogger(__name__)


@dataclass
class AgentEnvConfig:
    max_steps: int = 6
    max_search_calls: int = 4

    # Evidence memory
    deduplicate_evidence: bool = True
    max_evidence_passages: int = 20
    max_evidence_triples: int = 80
    max_memory_text_chars: int = 1200

    # If the agent never submits, use this answer.
    fallback_answer: str = "I don't know"

    # Finalization
    enable_finalization: bool = False
    preserve_base_evidence: bool = True

    # Debug
    save_trajectories: bool = False
    trajectory_dir: Optional[str] = None


def to_plain(obj: Any) -> Any:
    if obj is None:
        return None

    if is_dataclass(obj):
        return {k: to_plain(v) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback for objects with __dict__.
    if hasattr(obj, "__dict__"):
        return {k: to_plain(v) for k, v in vars(obj).items()}

    return str(obj)


def normalize_triple_key(triple: Any) -> str:
    """Build a stable key for deduplicating triples/facts."""
    plain = to_plain(triple)

    if isinstance(plain, dict):
        inner = plain.get("triple", plain)
        if isinstance(inner, dict):
            subj = inner.get("subject") or inner.get("subj") or inner.get("head") or ""
            pred = inner.get("predicate") or inner.get("relation") or inner.get("pred") or ""
            obj = inner.get("object") or inner.get("obj") or inner.get("tail") or ""
            return f"{subj}||{pred}||{obj}".lower()

        return json.dumps(plain, ensure_ascii=False, sort_keys=True).lower()

    if isinstance(plain, list):
        return json.dumps(plain, ensure_ascii=False).lower()

    return str(plain).lower()


def passage_to_memory_text(passage: RetrievedPassage, max_chars: int = 1200) -> str:
    """
    Convert a passage to compact memory text for the next search request.
    """
    pid = getattr(passage, "passage_id", "unknown")
    text = getattr(passage, "text", "") or ""
    text = text.strip()

    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."

    return f"[PASSAGE {pid}] {text}"


def triple_to_memory_text(triple: Any) -> str:
    """
    Convert candidate/filtered triple objects into compact text.
    """
    plain = to_plain(triple)

    if isinstance(plain, dict):
        inner = plain.get("triple", plain)

        if isinstance(inner, dict):
            subj = inner.get("subject") or inner.get("subj") or inner.get("head") or ""
            pred = inner.get("predicate") or inner.get("relation") or inner.get("pred") or ""
            obj = inner.get("object") or inner.get("obj") or inner.get("tail") or ""

            if subj or pred or obj:
                score = plain.get("score")
                if score is not None:
                    return f"[TRIPLE score={score}] ({subj}, {pred}, {obj})"
                return f"[TRIPLE] ({subj}, {pred}, {obj})"

        return f"[TRIPLE] {json.dumps(plain, ensure_ascii=False)[:500]}"

    return f"[TRIPLE] {str(plain)[:500]}"



class EvidenceMemory:
    """
    Stores evidence across multiple SearchGraph calls.
    It keeps:
        - retrieved passages
        - candidate triples
        - LLM-filtered triples
        - seed/search metadata
    """

    def __init__(
        self,
        deduplicate: bool = True,
        max_passages: int = 20,
        max_triples: int = 80,
        max_text_chars: int = 1200,
    ):
        self.deduplicate = deduplicate
        self.max_passages = max_passages
        self.max_triples = max_triples
        self.max_text_chars = max_text_chars

        self.passages: List[RetrievedPassage] = []
        self.candidate_triples: List[Any] = []
        self.filtered_triples: List[Any] = []
        self.seed_info: List[Any] = []

        self.seen_passage_ids = set()
        self.seen_candidate_triple_keys = set()
        self.seen_filtered_triple_keys = set()

    # Passage memory
    def add_many(self, passages: Sequence[RetrievedPassage]) -> None:
        for passage in passages:
            self.add(passage)

    def add(self, passage: RetrievedPassage) -> None:
        passage_id = getattr(passage, "passage_id", None)

        if self.deduplicate and passage_id in self.seen_passage_ids:
            return

        if len(self.passages) >= self.max_passages:
            return

        self.passages.append(passage)

        if passage_id is not None:
            self.seen_passage_ids.add(passage_id)

    # Triple memory

    def add_candidate_triples(self, triples: Sequence[Any]) -> None:
        for triple in triples:
            self.add_candidate_triple(triple)

    def add_candidate_triple(self, triple: Any) -> None:
        key = normalize_triple_key(triple)

        if self.deduplicate and key in self.seen_candidate_triple_keys:
            return

        if len(self.candidate_triples) >= self.max_triples:
            return

        self.candidate_triples.append(triple)
        self.seen_candidate_triple_keys.add(key)

    def add_filtered_triples(self, triples: Sequence[Any]) -> None:
        for triple in triples:
            self.add_filtered_triple(triple)

    def add_filtered_triple(self, triple: Any) -> None:
        key = normalize_triple_key(triple)

        if self.deduplicate and key in self.seen_filtered_triple_keys:
            return

        if len(self.filtered_triples) >= self.max_triples:
            return

        self.filtered_triples.append(triple)
        self.seen_filtered_triple_keys.add(key)

    # Search result memory

    def add_search_result(self, search_result: Any) -> None:
        """
        Add passages, candidate triples, filtered triples, and seed info from one SearchGraph result.
        """
        passages = getattr(search_result, "passages", None)
        if isinstance(passages, list):
            self.add_many(passages)

        candidate_triples = getattr(search_result, "candidate_triples", None)
        if isinstance(candidate_triples, list):
            self.add_candidate_triples(candidate_triples)

        filtered_triples = getattr(search_result, "filtered_triples", None)
        if isinstance(filtered_triples, list):
            self.add_filtered_triples(filtered_triples)

        seed_info = getattr(search_result, "seed_info", None)
        if seed_info is not None:
            self.seed_info.append(seed_info)



    def to_list(self) -> List[RetrievedPassage]:
        return list(self.passages)

    def passage_ids(self) -> List[str]:
        return [getattr(p, "passage_id", "") for p in self.passages]

    def texts(self) -> List[str]:
        return [getattr(p, "text", "") for p in self.passages]

    def memory_texts(self) -> List[str]:
        texts: List[str] = []

        for p in self.passages:
            texts.append(passage_to_memory_text(p, max_chars=self.max_text_chars))

        for t in self.filtered_triples[: self.max_triples]:
            texts.append(triple_to_memory_text(t))

        return texts

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "passages": [to_plain(p) for p in self.passages],
            "candidate_triples": [to_plain(t) for t in self.candidate_triples],
            "filtered_triples": [to_plain(t) for t in self.filtered_triples],
            "seed_info": [to_plain(s) for s in self.seed_info],
            "passage_ids": self.passage_ids(),
        }

@dataclass
class EpisodeFinalizationResult:
    base_passages: List[Dict[str, Any]]
    llm_selected_passages: List[Dict[str, Any]]
    fused_passages: List[Dict[str, Any]]

    predicted_answer: str
    supporting_passage_ids: List[str]
    supporting_triples: List[str]
    confidence: Optional[float]

    selector_metadata: Dict[str, Any]
    fusion_metadata: Dict[str, Any]
    answer_metadata: Dict[str, Any]

    raw_answer_response: Optional[str] = None


class AgentEnv:
    def __init__(
        self,
        config: AgentEnvConfig,
        reasoning_agent: ReasoningAgent,
        search_engine: PPRSearchEngine,
        evidence_selector: Optional[EvidenceSelectorV2] = None,
        evidence_fuser: Optional[HybridEvidenceFuser] = None,
        answer_reader: Optional[GroundedAnswerReader] = None,
    ):
        self.config = config
        self.reasoning_agent = reasoning_agent
        self.search_engine = search_engine
        # llm_select & hybrid func
        self.evidence_selector = evidence_selector
        self.evidence_fuser = evidence_fuser
        self.answer_reader = answer_reader

        if self.config.enable_finalization:
            if self.evidence_selector is None:
                raise ValueError(
                    "evidence_selector must be provided when "
                    "enable_finalization=True."
                    )

            if self.evidence_fuser is None:
                raise ValueError(
                    "evidence_fuser must be provided when "
                    "enable_finalization=True."
                    )

            if self.answer_reader is None:
                raise ValueError(
                    "answer_reader must be provided when "
                    "enable_finalization=True."
                    )
            
        if self.config.save_trajectories:
            if self.config.trajectory_dir is None:
                raise ValueError(
                    "trajectory_dir must be provided when save_trajectories=True."
                )
            os.makedirs(self.config.trajectory_dir, exist_ok=True)

    def run_episode(
        self,
        question: str,
        question_id: Optional[str] = None,
        gold_answers: Optional[Sequence[str]] = None,
        gold_passage_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentTrajectory:
        qid = question_id or "unknown"

        trajectory = AgentTrajectory(
            question_id=qid,
            question=question,
            gold_answers=list(gold_answers or []),
            gold_passage_ids=list(gold_passage_ids or []),
            metadata=dict(metadata or {}),
        )

        evidence_memory = EvidenceMemory(
            deduplicate=self.config.deduplicate_evidence,
            max_passages=self.config.max_evidence_passages,
            max_triples=self.config.max_evidence_triples,
            max_text_chars=self.config.max_memory_text_chars,
        )

        num_search_calls = 0
        forced_stop = False
        requested_action_before_forced_stop: Optional[Dict[str, Any]] = None
        termination_reason: Optional[str] = None

        for step_id in range(self.config.max_steps):
            action = self.reasoning_agent.decide(
                question=question,
                question_id=qid,
                step_id=step_id,
                evidence_memory=evidence_memory,
            )

            if (
                action.action == "SearchGraph"
                and num_search_calls >= self.config.max_search_calls
            ):
                forced_stop = True
                requested_action_before_forced_stop = to_plain(action)
                termination_reason = "search_budget_exhausted"

                action = AgentAction(
                    action="SubmitFinalAnswer",
                    answer=self.config.fallback_answer,
                    reasoning_summary=(
                        "Search budget exhausted before the agent submitted an answer."
                    ),
                    raw_output=action.raw_output,
                )

            action.validate()

            step = AgentStep(
                step_id=step_id,
                action=action,
                observation=None,
            )

            if action.action == "SubmitFinalAnswer":
                trajectory.final_answer = action.answer

                if termination_reason is None:
                    termination_reason = "controller_submitted_answer"

                step.observation = AgentObservation(
                    step_id=step_id,
                    message="Agent submitted final answer.",
                    evidence_memory=evidence_memory.to_list(),
                )

                trajectory.steps.append(step)
                break

            if action.action == "SearchGraph":
                assert action.search_request is not None

                search_request = self._prepare_search_request(
                    action.search_request,
                    question=question,
                    step_id=step_id,
                    evidence_memory=evidence_memory,
                )

                search_result = self.search_engine.search(search_request)
                num_search_calls += 1
                evidence_memory.add_search_result(search_result)

                step.observation = AgentObservation(
                    step_id=step_id,
                    search_result=search_result,
                    message="SearchGraph executed.",
                    evidence_memory=evidence_memory.to_list(),
                )

                trajectory.steps.append(step)
                continue

            raise ValueError(f"Unknown action type: {action.action}")

        if trajectory.final_answer is None:
            trajectory.final_answer = self.config.fallback_answer
            termination_reason = termination_reason or "max_steps_reached"

            final_step_id = len(trajectory.steps)
            final_action = AgentAction(
                action="SubmitFinalAnswer",
                answer=self.config.fallback_answer,
                reasoning_summary="Max steps reached without final answer.",
                raw_output=None,
            )
            final_observation = AgentObservation(
                step_id=final_step_id,
                message="Max steps reached. Fallback answer used.",
                evidence_memory=evidence_memory.to_list(),
            )
            trajectory.steps.append(
                AgentStep(
                    step_id=final_step_id,
                    action=final_action,
                    observation=final_observation,
                )
            )

        memory_payload = evidence_memory.to_serializable()
        finalization_result: Optional[EpisodeFinalizationResult] = None

        if self.config.enable_finalization:
            finalization_result = self.finalize_episode(
                question=question,
                evidence_memory=evidence_memory,
            )

        trajectory.metadata.update(
            {
                "num_search_calls": num_search_calls,
                "num_steps": len(trajectory.steps),
                "retrieved_passage_ids": memory_payload["passage_ids"],
                "num_evidence_passages": len(memory_payload["passages"]),
                "num_candidate_triples": len(memory_payload["candidate_triples"]),
                "num_filtered_triples": len(memory_payload["filtered_triples"]),
                "forced_stop": forced_stop,
                "termination_reason": termination_reason,
            }
        )

        if requested_action_before_forced_stop is not None:
            trajectory.metadata[
                "requested_action_before_forced_stop"
            ] = requested_action_before_forced_stop

        trajectory.metadata["_evidence_payload"] = memory_payload

        if finalization_result is not None:
            trajectory.metadata["finalization"] = {
                "selector": finalization_result.selector_metadata,
                "fusion": finalization_result.fusion_metadata,
                "answer_reader": finalization_result.answer_metadata,
            }

            if self.config.preserve_base_evidence:
                trajectory.metadata[
                    "_base_evidence_passages"
                ] = finalization_result.base_passages

            trajectory.metadata[
                "_llm_selected_passages"
            ] = finalization_result.llm_selected_passages
            trajectory.metadata[
                "_fused_evidence_passages"
            ] = finalization_result.fused_passages
            trajectory.metadata[
                "reader_predicted_answer"
            ] = finalization_result.predicted_answer
            trajectory.metadata[
                "reader_supporting_passage_ids"
            ] = finalization_result.supporting_passage_ids
            trajectory.metadata[
                "reader_supporting_triples"
            ] = finalization_result.supporting_triples
            trajectory.metadata[
                "reader_confidence"
            ] = finalization_result.confidence
            trajectory.metadata[
                "reader_raw_response"
            ] = finalization_result.raw_answer_response

        if self.config.save_trajectories:
            self.save_trajectory(trajectory)

        return trajectory


    def run_batch(
        self,
        examples: Sequence[Dict[str, Any]],
        question_key: str = "question",
        id_key: str = "id",
        answers_key: str = "answer",
        gold_passage_ids_key: str = "gold_passage_ids",
    ) -> List[AgentTrajectory]:
       
        trajectories: List[AgentTrajectory] = []

        for ex in tqdm(examples, desc="Agent retrieval"):
            question = ex[question_key]
            question_id = str(ex.get(id_key, len(trajectories)))

            gold_answers = ex.get(answers_key, [])
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            gold_passage_ids = ex.get(gold_passage_ids_key, [])

            trajectory = self.run_episode(
                question=question,
                question_id=question_id,
                gold_answers=gold_answers,
                gold_passage_ids=gold_passage_ids,
                metadata={
                    "raw_example": ex,
                },
            )
            trajectories.append(trajectory)

        return trajectories

  
    def _prepare_search_request(
        self,
        request: SearchGraphRequest,
        question: str,
        step_id: int,
        evidence_memory: EvidenceMemory,
    ) -> SearchGraphRequest:
        request.question = question
        request.step_id = step_id

        request.evidence_so_far = evidence_memory.memory_texts()

        return request

    def finalize_episode(
            self, question: str, evidence_memory: EvidenceMemory,
            ) -> EpisodeFinalizationResult:
        if self.evidence_selector is None:
            raise RuntimeError("evidence_selector is not configured.")

        if self.evidence_fuser is None:
            raise RuntimeError("evidence_fuser is not configured.")

        if self.answer_reader is None:
            raise RuntimeError("answer_reader is not configured.")

        memory_payload = evidence_memory.to_serializable()

        base_passages = list(memory_payload.get("passages") or [])

        filtered_triples = list(memory_payload.get("filtered_triples") or [])

        selector_result = self.evidence_selector.select(
            question=question,
            passages=base_passages,
            filtered_triples=filtered_triples,
        )

        fusion_result = self.evidence_fuser.merge(
            base_passages=base_passages,
            llm_passages=selector_result.passages,
        )

        answer_result = self.answer_reader.generate(
            question=question,
            passages=fusion_result.passages,
            filtered_triples=filtered_triples,
        )

        return EpisodeFinalizationResult(
            base_passages=base_passages,
            llm_selected_passages=(selector_result.passages),
            fused_passages=fusion_result.passages,
            predicted_answer=(answer_result.predicted_answer),
            supporting_passage_ids=(answer_result.supporting_passage_ids),
            supporting_triples=(answer_result.supporting_triples),
            confidence=answer_result.confidence,
            selector_metadata=(selector_result.metadata()),
            fusion_metadata=dict(fusion_result.metadata),
            answer_metadata=dict(answer_result.metadata),
            raw_answer_response=(answer_result.raw_response),
        )


    def save_trajectory(self, trajectory: AgentTrajectory) -> str:
        if self.config.trajectory_dir is None:
            raise ValueError("trajectory_dir is None.")

        filename = f"{trajectory.question_id}.json"
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        path = os.path.join(self.config.trajectory_dir, safe_filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(trajectory_to_dict(trajectory), f, ensure_ascii=False, indent=2)

        logger.info("Saved trajectory to %s", path)
        return path


def collect_evidence_from_steps(trajectory_dict: Dict[str, Any]) -> Dict[str, Any]:
    passages: List[Any] = []
    candidate_triples: List[Any] = []
    filtered_triples: List[Any] = []
    seed_info: List[Any] = []

    seen_passage_ids = set()
    seen_candidate_triples = set()
    seen_filtered_triples = set()

    for step in trajectory_dict.get("steps", []):
        obs = step.get("observation")
        if not isinstance(obs, dict):
            continue

        search_result = obs.get("search_result")
        if not isinstance(search_result, dict):
            continue

        for p in search_result.get("passages", []) or []:
            if isinstance(p, dict):
                pid = p.get("passage_id") or p.get("id") or p.get("idx")
            else:
                pid = None

            key = str(pid) if pid is not None else json.dumps(to_plain(p), ensure_ascii=False)[:200]

            if key in seen_passage_ids:
                continue

            seen_passage_ids.add(key)
            passages.append(p)

        for t in search_result.get("candidate_triples", []) or []:
            key = normalize_triple_key(t)
            if key in seen_candidate_triples:
                continue

            seen_candidate_triples.add(key)
            candidate_triples.append(t)

        for t in search_result.get("filtered_triples", []) or []:
            key = normalize_triple_key(t)
            if key in seen_filtered_triples:
                continue

            seen_filtered_triples.add(key)
            filtered_triples.append(t)

        if search_result.get("seed_info") is not None:
            seed_info.append(search_result.get("seed_info"))

    return {
        "passages": passages,
        "candidate_triples": candidate_triples,
        "filtered_triples": filtered_triples,
        "seed_info": seed_info,
        "passage_ids": [
            str(p.get("passage_id") or p.get("id") or p.get("idx"))
            for p in passages
            if isinstance(p, dict)
        ],
    }


def trajectory_to_dict(trajectory: AgentTrajectory) -> Dict[str, Any]:
    data = to_plain(trajectory)

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata"] = metadata

    memory_payload = metadata.get("_evidence_payload")

    if not isinstance(memory_payload, dict):
        memory_payload = collect_evidence_from_steps(data)

    step_payload = collect_evidence_from_steps(data)

    base_memory_passages = (
        memory_payload.get("passages")
        or step_payload.get("passages")
        or []
    )
    candidate_triples = (
        memory_payload.get("candidate_triples")
        or step_payload.get("candidate_triples")
        or []
    )
    filtered_triples = (
        memory_payload.get("filtered_triples")
        or step_payload.get("filtered_triples")
        or []
    )
    seed_info = (
        memory_payload.get("seed_info")
        or step_payload.get("seed_info")
        or []
    )

    base_evidence_passages = metadata.get("_base_evidence_passages")
    llm_selected_passages = metadata.get("_llm_selected_passages")
    fused_evidence_passages = metadata.get("_fused_evidence_passages")

    if not isinstance(base_evidence_passages, list):
        base_evidence_passages = base_memory_passages

    data["base_evidence_passages"] = base_evidence_passages

    if isinstance(llm_selected_passages, list):
        data["llm_selected_passages"] = llm_selected_passages

    if isinstance(fused_evidence_passages, list):
        data["fused_evidence_passages"] = fused_evidence_passages
        data["evidence_passages"] = fused_evidence_passages
    else:
        data["evidence_passages"] = base_memory_passages

    data["candidate_triples"] = candidate_triples
    data["filtered_triples"] = filtered_triples
    data["seed_info"] = seed_info

    data["controller_final_answer"] = data.get("final_answer")

    reader_predicted_answer = metadata.get("reader_predicted_answer")
    if reader_predicted_answer is not None:
        data["predicted_answer"] = reader_predicted_answer

    if "reader_supporting_passage_ids" in metadata:
        data["supporting_passage_ids"] = metadata[
            "reader_supporting_passage_ids"
        ]

    if "reader_supporting_triples" in metadata:
        data["supporting_triples"] = metadata[
            "reader_supporting_triples"
        ]

    if "reader_confidence" in metadata:
        data["confidence"] = metadata["reader_confidence"]

    if "reader_raw_response" in metadata:
        data["raw_response"] = metadata["reader_raw_response"]

    final_evidence = data.get("evidence_passages") or []
    metadata["retrieved_passage_ids"] = [
        str(p.get("passage_id") or p.get("id") or p.get("idx"))
        for p in final_evidence
        if isinstance(p, dict)
    ]
    metadata["num_evidence_passages"] = len(final_evidence)
    metadata["num_candidate_triples"] = len(candidate_triples)
    metadata["num_filtered_triples"] = len(filtered_triples)

    metadata.pop("_evidence_payload", None)
    metadata.pop("_base_evidence_passages", None)
    metadata.pop("_llm_selected_passages", None)
    metadata.pop("_fused_evidence_passages", None)

    return data


def save_trajectories_jsonl(
    path: str,
    trajectories: Sequence[AgentTrajectory],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory_to_dict(trajectory), ensure_ascii=False) + "\n")


def load_examples_json(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "examples", "queries", "questions"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    raise ValueError(f"Could not load examples from {path}")


def run_single_agent_episode(
    question: str,
    reasoning_agent: ReasoningAgent,
    search_engine: PPRSearchEngine,
    question_id: str = "debug",
    max_steps: int = 6,
    max_search_calls: int = 4,
) -> AgentTrajectory:
    env = AgentEnv(
        config=AgentEnvConfig(
            max_steps=max_steps,
            max_search_calls=max_search_calls,
        ),
        reasoning_agent=reasoning_agent,
        search_engine=search_engine,
    )

    return env.run_episode(
        question=question,
        question_id=question_id,
    )