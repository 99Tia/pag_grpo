from __future__ import annotations
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from .agent_env import AgentEnv, EvidenceMemory, to_plain
from .grpo_policy import GRPOPolicy
from .grpo_types import (
    GoldSupportingPassage,
    PolicyStepSample,
    RolloutGroup,
    RolloutTrajectory,
)
from .schema import SearchGraphRequest
from .trajectory_reward import TrajectoryRewardCalculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GRPORolloutExample:
    question_id: str
    question: str
    gold_answers: Sequence[str] = field(default_factory=tuple)
    gold_supporting_passages: Sequence[GoldSupportingPassage] = field(
        default_factory=tuple
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.question_id).strip():
            raise ValueError("question_id must be non-empty.")
        if not str(self.question).strip():
            raise ValueError("question must be non-empty.")

        for index, passage in enumerate(self.gold_supporting_passages):
            if not isinstance(passage, GoldSupportingPassage):
                raise TypeError(
                    "gold_supporting_passages must contain "
                    f"GoldSupportingPassage objects; item {index} is "
                    f"{type(passage).__name__}."
                )
            passage.validate()

    @classmethod
    def from_mapping(
        cls,
        example: Mapping[str, Any],
        *,
        question_key: str = "question",
        id_key: str = "id",
        answers_key: Optional[str] = None,
        supporting_passages_key: Optional[str] = None,
        paragraphs_key: str = "paragraphs",
        keep_raw_example_in_metadata: bool = False,
    ) -> "GRPORolloutExample":
        if question_key not in example:
            raise KeyError(f"Missing question field {question_key!r}.")

        question = str(example[question_key]).strip()
        question_id = str(
            example.get(id_key, example.get("question_id", "unknown"))
        ).strip()

        if answers_key is not None:
            raw_answers = example.get(answers_key, [])
        else:
            raw_answers = example.get(
                "answer",
                example.get("answers", example.get("gold_answers", [])),
            )
        gold_answers = _coerce_string_list(raw_answers)

        if supporting_passages_key is not None:
            raw_supports = example.get(supporting_passages_key, [])
        elif "gold_supporting_passages" in example:
            raw_supports = example.get("gold_supporting_passages", [])
        elif "supporting_passages" in example:
            raw_supports = example.get("supporting_passages", [])
        else:
            paragraphs = example.get(paragraphs_key, [])
            if isinstance(paragraphs, Sequence) and not isinstance(
                paragraphs, (str, bytes)
            ):
                raw_supports = [
                    paragraph
                    for paragraph in paragraphs
                    if _mapping_get(paragraph, "is_supporting", False)
                ]
            else:
                raw_supports = []

        gold_supports = _coerce_gold_supporting_passages(raw_supports)

        metadata: Dict[str, Any] = {
            "source_fields": {
                "question_key": question_key,
                "id_key": id_key,
                "answers_key": answers_key,
                "supporting_passages_key": supporting_passages_key,
                "paragraphs_key": paragraphs_key,
            }
        }
        source_metadata = example.get("metadata")
        if isinstance(source_metadata, Mapping):
            metadata["source_metadata"] = to_plain(source_metadata)
        if keep_raw_example_in_metadata:
            metadata["raw_example"] = to_plain(example)

        return cls(
            question_id=question_id,
            question=question,
            gold_answers=tuple(gold_answers),
            gold_supporting_passages=tuple(gold_supports),
            metadata=metadata,
        )


@dataclass(frozen=True)
class GRPORolloutConfig:
    group_size: int = 4
    base_seed: int = 42
    sequential_groups: bool = True
    tokenize_observations: bool = False
    max_observation_passage_ids: int = 10
    score_trajectories: bool = True
    compute_advantages: bool = True
    raise_on_search_error: bool = True
    raise_on_finalization_error: bool = True
    validate_records: bool = True

    def __post_init__(self) -> None:
        if self.group_size <= 1:
            raise ValueError(
                "group_size must be greater than one for group-relative GRPO."
            )
        if self.max_observation_passage_ids < 0:
            raise ValueError("max_observation_passage_ids must be non-negative.")
        if not self.sequential_groups:
            raise ValueError(
                "Parallel rollout collection is intentionally disabled in this "
                "first implementation. Use sequential_groups=True."
            )


@dataclass
class RolloutCollectionSummary:
    num_groups: int = 0
    num_trajectories: int = 0
    num_policy_steps: int = 0
    num_policy_tokens: int = 0
    num_search_calls: int = 0
    num_forced_stops: int = 0
    num_zero_variance_groups: int = 0
    elapsed_seconds: float = 0.0

    @property
    def zero_variance_rate(self) -> float:
        if self.num_groups == 0:
            return 0.0
        return self.num_zero_variance_groups / self.num_groups

    @property
    def mean_trajectories_per_group(self) -> float:
        if self.num_groups == 0:
            return 0.0
        return self.num_trajectories / self.num_groups

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_groups": self.num_groups,
            "num_trajectories": self.num_trajectories,
            "num_policy_steps": self.num_policy_steps,
            "num_policy_tokens": self.num_policy_tokens,
            "num_search_calls": self.num_search_calls,
            "num_forced_stops": self.num_forced_stops,
            "num_zero_variance_groups": self.num_zero_variance_groups,
            "zero_variance_rate": self.zero_variance_rate,
            "mean_trajectories_per_group": self.mean_trajectories_per_group,
            "elapsed_seconds": self.elapsed_seconds,
        }


class GRPORolloutCollector:
    def __init__(
        self,
        *,
        policy: GRPOPolicy,
        environment: AgentEnv,
        reward_calculator: TrajectoryRewardCalculator,
        config: Optional[GRPORolloutConfig] = None,
    ) -> None:
        self.policy = policy
        self.environment = environment
        self.reward_calculator = reward_calculator
        self.config = config or GRPORolloutConfig()

        if self.environment.config.max_steps <= 0:
            raise ValueError("AgentEnv max_steps must be greater than zero.")
        if self.environment.config.max_search_calls <= 0:
            raise ValueError(
                "AgentEnv max_search_calls must be greater than zero."
            )

        reward_max_searches = self.reward_calculator.config.max_search_calls
        env_max_searches = self.environment.config.max_search_calls
        if reward_max_searches != env_max_searches:
            raise ValueError(
                "Reward and environment search budgets must match: "
                f"reward={reward_max_searches}, environment={env_max_searches}."
            )

    def collect_group(
        self,
        example: GRPORolloutExample,
        *,
        group_size: Optional[int] = None,
        rollout_iteration: int = 0,
    ) -> RolloutGroup:
        size = self.config.group_size if group_size is None else int(group_size)
        if size <= 1:
            raise ValueError("group_size must be greater than one.")
        if rollout_iteration < 0:
            raise ValueError("rollout_iteration must be non-negative.")

        group = RolloutGroup(
            question_id=example.question_id,
            question=example.question,
            epsilon=self.reward_calculator.config.advantage_epsilon,
            metadata={
                "rollout_iteration": rollout_iteration,
                "requested_group_size": size,
                "base_seed": self.config.base_seed,
                "example_metadata": to_plain(example.metadata),
            },
        )

        start = time.perf_counter()
        for group_index in range(size):
            trajectory_seed = derive_rollout_seed(
                base_seed=self.config.base_seed,
                question_id=example.question_id,
                rollout_iteration=rollout_iteration,
                group_index=group_index,
                step_id=None,
            )
            trajectory = self.collect_trajectory(
                example,
                group_index=group_index,
                rollout_iteration=rollout_iteration,
                trajectory_seed=trajectory_seed,
            )
            group.add_trajectory(trajectory)

        if self.config.score_trajectories:
            self.reward_calculator.score_group(
                group,
                compute_advantages=self.config.compute_advantages,
            )

        group.metadata.update(
            {
                "elapsed_seconds": time.perf_counter() - start,
                "reward_totals": [
                    float(trajectory.reward.total)
                    for trajectory in group.trajectories
                ],
                "advantages": [trajectory.advantage for trajectory in group.trajectories],
                "zero_variance": group.zero_variance,
            }
        )

        if self.config.validate_records:
            group.validate(require_old_log_probs=True)

        return group

    def collect_trajectory(
        self,
        example: GRPORolloutExample,
        *,
        group_index: int,
        rollout_iteration: int = 0,
        trajectory_seed: Optional[int] = None,
    ) -> RolloutTrajectory:
        if group_index < 0:
            raise ValueError("group_index must be non-negative.")
        if rollout_iteration < 0:
            raise ValueError("rollout_iteration must be non-negative.")

        if trajectory_seed is None:
            trajectory_seed = derive_rollout_seed(
                base_seed=self.config.base_seed,
                question_id=example.question_id,
                rollout_iteration=rollout_iteration,
                group_index=group_index,
                step_id=None,
            )

        trajectory_id = (
            f"{example.question_id}::iter-{rollout_iteration}::"
            f"group-{group_index}::seed-{trajectory_seed}"
        )
        trajectory = RolloutTrajectory(
            trajectory_id=trajectory_id,
            question_id=example.question_id,
            question=example.question,
            group_index=group_index,
            gold_answers=list(example.gold_answers),
            gold_supporting_passages=list(example.gold_supporting_passages),
            metadata={
                "rollout_iteration": rollout_iteration,
                "trajectory_seed": trajectory_seed,
                "environment": {
                    "max_steps": self.environment.config.max_steps,
                    "max_search_calls": self.environment.config.max_search_calls,
                    "enable_finalization": self.environment.config.enable_finalization,
                },
            },
        )

        evidence_memory = EvidenceMemory(
            deduplicate=self.environment.config.deduplicate_evidence,
            max_passages=self.environment.config.max_evidence_passages,
            max_triples=self.environment.config.max_evidence_triples,
            max_text_chars=self.environment.config.max_memory_text_chars,
        )
        memory_identity = id(evidence_memory)

        start = time.perf_counter()
        num_search_calls = 0
        termination_reason: Optional[str] = None
        controller_final_answer: Optional[str] = None
        environment_errors: List[Dict[str, Any]] = []

        for step_id in range(self.environment.config.max_steps):
            step_seed = derive_rollout_seed(
                base_seed=trajectory_seed,
                question_id=example.question_id,
                rollout_iteration=rollout_iteration,
                group_index=group_index,
                step_id=step_id,
            )

            sampled = self.policy.sample_action(
                question=example.question,
                question_id=example.question_id,
                step_id=step_id,
                evidence_memory=evidence_memory,
                seed=step_seed,
            )
            action = sampled.agent_action
            record = sampled.rollout_action
            action.validate()

            if (
                action.action == "SearchGraph"
                and num_search_calls >= self.environment.config.max_search_calls
            ):
                record.forced_stop = True
                controller_final_answer = self.environment.config.fallback_answer
                termination_reason = "search_budget_exhausted"
                self.policy.attach_observation(
                    rollout_action=record,
                    observation_text=(
                        "SearchGraph was not executed because the trajectory "
                        "search budget was exhausted."
                    ),
                    observation_payload={
                        "step_id": step_id,
                        "message": "Search budget exhausted.",
                        "requested_action": to_plain(action),
                        "num_search_calls": num_search_calls,
                        "max_search_calls": self.environment.config.max_search_calls,
                    },
                    tokenize_observation=self.config.tokenize_observations,
                )
                trajectory.add_action(record)
                break

            if action.action == "SubmitFinalAnswer":
                controller_final_answer = str(action.answer or "").strip()
                termination_reason = "controller_submitted_answer"
                self.policy.attach_observation(
                    rollout_action=record,
                    observation_text="Controller submitted its final answer.",
                    observation_payload={
                        "step_id": step_id,
                        "message": "Controller submitted final answer.",
                        "answer": controller_final_answer,
                    },
                    tokenize_observation=self.config.tokenize_observations,
                )
                trajectory.add_action(record)
                break

            if action.action != "SearchGraph":
                raise ValueError(f"Unknown controller action: {action.action!r}.")

            assert action.search_request is not None
            search_request = self._prepare_search_request(
                request=action.search_request,
                question=example.question,
                step_id=step_id,
                evidence_memory=evidence_memory,
            )

            try:
                search_result = self.environment.search_engine.search(search_request)
            except Exception as exc:
                error_payload = {
                    "stage": "search",
                    "step_id": step_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                environment_errors.append(error_payload)
                if self.config.raise_on_search_error:
                    raise RuntimeError(
                        f"SearchGraph failed for trajectory {trajectory_id!r}, "
                        f"step {step_id}."
                    ) from exc

                record.forced_stop = True
                controller_final_answer = self.environment.config.fallback_answer
                termination_reason = "search_error"
                self.policy.attach_observation(
                    rollout_action=record,
                    observation_text="SearchGraph failed; trajectory stopped.",
                    observation_payload=error_payload,
                    tokenize_observation=self.config.tokenize_observations,
                )
                trajectory.add_action(record)
                break

            num_search_calls += 1
            evidence_memory.add_search_result(search_result)

            observation_payload = {
                "step_id": step_id,
                "message": "SearchGraph executed.",
                "search_result": to_plain(search_result),
                "evidence_memory": evidence_memory.to_serializable(),
            }
            observation_text = format_search_observation(
                search_result,
                max_passage_ids=self.config.max_observation_passage_ids,
            )
            self.policy.attach_observation(
                rollout_action=record,
                observation_text=observation_text,
                observation_payload=observation_payload,
                tokenize_observation=self.config.tokenize_observations,
            )
            trajectory.add_action(record)

        if controller_final_answer is None:
            controller_final_answer = self.environment.config.fallback_answer
            termination_reason = termination_reason or "max_steps_reached"

        trajectory.controller_final_answer = controller_final_answer
        trajectory.num_search_calls = num_search_calls
        trajectory.termination_reason = termination_reason
        trajectory.forced_stop = bool(
            any(action.forced_stop for action in trajectory.actions)
        )

        memory_payload = evidence_memory.to_serializable()
        trajectory.base_evidence_passages = list(
            memory_payload.get("passages") or []
        )
        trajectory.filtered_triples = list(
            memory_payload.get("filtered_triples") or []
        )

        if self.environment.config.enable_finalization:
            try:
                finalization = self.environment.finalize_episode(
                    question=example.question,
                    evidence_memory=evidence_memory,
                )
            except Exception as exc:
                error_payload = {
                    "stage": "finalization",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                environment_errors.append(error_payload)
                if self.config.raise_on_finalization_error:
                    raise RuntimeError(
                        f"Finalization failed for trajectory {trajectory_id!r}."
                    ) from exc
            else:
                trajectory.base_evidence_passages = list(
                    finalization.base_passages
                )
                trajectory.selected_evidence_passages = list(
                    finalization.llm_selected_passages
                )
                trajectory.fused_evidence_passages = list(
                    finalization.fused_passages
                )
                trajectory.reader_predicted_answer = str(
                    finalization.predicted_answer or ""
                )
                trajectory.reader_supporting_passage_ids = list(
                    finalization.supporting_passage_ids
                )
                trajectory.reader_supporting_triples = list(
                    finalization.supporting_triples
                )
                trajectory.metadata["finalization"] = {
                    "selector": to_plain(finalization.selector_metadata),
                    "fusion": to_plain(finalization.fusion_metadata),
                    "answer_reader": to_plain(finalization.answer_metadata),
                    "confidence": finalization.confidence,
                    "raw_answer_response": finalization.raw_answer_response,
                }

        trajectory.metadata.update(
            {
                "elapsed_seconds": time.perf_counter() - start,
                "memory_identity": memory_identity,
                "num_steps": trajectory.num_steps,
                "num_policy_tokens": trajectory.num_policy_tokens,
                "num_search_calls": num_search_calls,
                "num_evidence_passages": len(trajectory.base_evidence_passages),
                "num_filtered_triples": len(trajectory.filtered_triples),
                "termination_reason": termination_reason,
                "forced_stop": trajectory.forced_stop,
                "environment_errors": environment_errors,
            }
        )

        if self.config.validate_records:
            trajectory.validate(require_old_log_probs=True)

        return trajectory

    def collect_groups(
        self,
        examples: Iterable[GRPORolloutExample],
        *,
        rollout_iteration: int = 0,
        max_groups: Optional[int] = None,
        on_group_complete: Optional[
            Callable[[RolloutGroup, RolloutCollectionSummary], None]
        ] = None,
    ) -> tuple[List[RolloutGroup], RolloutCollectionSummary]:
        if max_groups is not None and max_groups <= 0:
            raise ValueError("max_groups must be positive or None.")

        groups: List[RolloutGroup] = []
        summary = RolloutCollectionSummary()
        start = time.perf_counter()

        for example in examples:
            if max_groups is not None and len(groups) >= max_groups:
                break

            group = self.collect_group(
                example,
                rollout_iteration=rollout_iteration,
            )
            groups.append(group)
            _update_collection_summary(summary, group)
            summary.elapsed_seconds = time.perf_counter() - start

            if on_group_complete is not None:
                on_group_complete(group, summary)

        summary.elapsed_seconds = time.perf_counter() - start
        return groups, summary

    def _prepare_search_request(
        self,
        *,
        request: SearchGraphRequest,
        question: str,
        step_id: int,
        evidence_memory: EvidenceMemory,
    ) -> SearchGraphRequest:
        return self.environment._prepare_search_request(  # noqa: SLF001
            request,
            question=question,
            step_id=step_id,
            evidence_memory=evidence_memory,
        )


def derive_rollout_seed(
    *,
    base_seed: int,
    question_id: str,
    rollout_iteration: int,
    group_index: int,
    step_id: Optional[int],
) -> int:
    payload = (
        f"{int(base_seed)}|{question_id}|{int(rollout_iteration)}|"
        f"{int(group_index)}|{step_id if step_id is not None else 'trajectory'}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


def format_search_observation(
    search_result: Any,
    *,
    max_passage_ids: int = 10,
) -> str:
    passages = list(getattr(search_result, "passages", None) or [])
    candidate_triples = list(
        getattr(search_result, "candidate_triples", None) or []
    )
    filtered_triples = list(
        getattr(search_result, "filtered_triples", None) or []
    )

    passage_ids: List[str] = []
    for passage in passages[:max_passage_ids]:
        passage_id = getattr(passage, "passage_id", None)
        if passage_id is None and isinstance(passage, Mapping):
            passage_id = passage.get("passage_id", passage.get("id"))
        if passage_id is not None:
            passage_ids.append(str(passage_id))

    parts = [
        "SearchGraph executed.",
        f"Retrieved passages: {len(passages)}.",
        f"Candidate triples: {len(candidate_triples)}.",
        f"Filtered triples: {len(filtered_triples)}.",
    ]
    if passage_ids:
        parts.append("Top passage IDs: " + ", ".join(passage_ids) + ".")
    return " ".join(parts)


def collect_policy_step_samples(
    groups: Sequence[RolloutGroup],
) -> List[PolicyStepSample]:
    samples: List[PolicyStepSample] = []
    for group in groups:
        samples.extend(group.to_policy_step_samples())
    return samples


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        value = [value]

    output: List[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _coerce_gold_supporting_passages(
    values: Any,
) -> List[GoldSupportingPassage]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, Mapping, GoldSupportingPassage)):
        values = [values]
    if not isinstance(values, Sequence):
        values = [values]

    output: List[GoldSupportingPassage] = []
    seen = set()
    for item in values:
        if isinstance(item, GoldSupportingPassage):
            passage = item
        elif isinstance(item, Mapping) or hasattr(item, "__dict__"):
            title = str(_mapping_get(item, "title", "") or "").strip()
            text = str(
                _mapping_get(
                    item,
                    "text",
                    _mapping_get(item, "passage", _mapping_get(item, "content", "")),
                )
                or ""
            ).strip()
            passage_id_value = _mapping_get(
                item,
                "passage_id",
                _mapping_get(item, "id", _mapping_get(item, "idx", None)),
            )
            metadata = _mapping_get(item, "metadata", {})
            passage = GoldSupportingPassage(
                title=title,
                text=text,
                passage_id=(
                    None if passage_id_value is None else str(passage_id_value)
                ),
                metadata=(
                    to_plain(metadata) if isinstance(metadata, Mapping) else {}
                ),
            )
        else:
            passage = GoldSupportingPassage(text=str(item).strip())

        passage.validate()
        key = (passage.title.casefold(), passage.text.casefold())
        if key in seen:
            continue
        seen.add(key)
        output.append(passage)

    return output


def _update_collection_summary(
    summary: RolloutCollectionSummary,
    group: RolloutGroup,
) -> None:
    summary.num_groups += 1
    summary.num_trajectories += group.group_size
    summary.num_policy_steps += sum(
        trajectory.num_steps for trajectory in group.trajectories
    )
    summary.num_policy_tokens += sum(
        trajectory.num_policy_tokens for trajectory in group.trajectories
    )
    summary.num_search_calls += sum(
        trajectory.num_search_calls for trajectory in group.trajectories
    )
    summary.num_forced_stops += sum(
        int(trajectory.forced_stop) for trajectory in group.trajectories
    )
    summary.num_zero_variance_groups += int(bool(group.zero_variance))


__all__ = [
    "GRPORolloutExample",
    "GRPORolloutConfig",
    "RolloutCollectionSummary",
    "GRPORolloutCollector",
    "derive_rollout_seed",
    "format_search_observation",
    "collect_policy_step_samples",
]
