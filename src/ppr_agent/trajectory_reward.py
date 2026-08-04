from __future__ import annotations
import math
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from .grpo_types import (
    GoldSupportingPassage,
    RolloutAction,
    RolloutGroup,
    RolloutTrajectory,
    TrajectoryReward,
)

@dataclass(frozen=True)
class TrajectoryRewardConfig:
    support_top_k: int = 5
    max_search_calls: int = 4
    allow_title_only_match: bool = True
    title_only_requires_unique: bool = True
    answer_f1_weight: float = 2.0
    support_recall_weight: float = 1.0
    full_support_weight: float = 1.0
    format_validity_weight: float = 0.10
    evidence_novelty_weight: float = 0.10
    #penalty weights
    search_cost_weight: float = -0.05
    duplicate_search_weight: float = -0.15
    forced_stop_weight: float = -0.10
    unknown_answer_weight: float = -0.10
    #Diagnostic-only metrics
    answer_exact_match_weight: float = 0.0
    support_precision_weight: float = 0.0
    advantage_epsilon: float = 1e-4
    zero_variance_threshold: float = 1e-8
    unknown_answers: Tuple[str, ...] = (
        "i don't know",
        "i do not know",
        "unknown",
        "not enough information",
        "insufficient information",
        "cannot determine",
    )

    def __post_init__(self) -> None:
        if self.support_top_k <= 0:
            raise ValueError("support_top_k must be greater than zero.")
        if self.max_search_calls <= 0:
            raise ValueError("max_search_calls must be greater than zero.")
        if self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be greater than zero.")
        if self.zero_variance_threshold < 0:
            raise ValueError("zero_variance_threshold must be non-negative.")

        for name, value in self.component_weights().items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Reward weight {name!r} must be finite.")

    def component_weights(self) -> Dict[str, float]:
        k = self.support_top_k
        return {
            "answer_f1": self.answer_f1_weight,
            "answer_exact_match": self.answer_exact_match_weight,
            f"support_recall_at_{k}": self.support_recall_weight,
            f"support_precision_at_{k}": self.support_precision_weight,
            f"full_support_at_{k}": self.full_support_weight,
            "format_validity": self.format_validity_weight,
            "evidence_novelty": self.evidence_novelty_weight,
            "search_cost": self.search_cost_weight,
            "duplicate_search_rate": self.duplicate_search_weight,
            "forced_stop": self.forced_stop_weight,
            "unknown_answer": self.unknown_answer_weight,
        }


@dataclass
class SupportMatchResult:
    top_k: int
    num_gold: int
    num_retrieved: int
    matched_gold_indices: List[int] = field(default_factory=list)
    matched_retrieved_indices: List[int] = field(default_factory=list)
    match_modes: List[str] = field(default_factory=list)
    recall: float = 0.0
    precision: float = 0.0
    full_support: float = 0.0
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "num_gold": self.num_gold,
            "num_retrieved": self.num_retrieved,
            "matched_gold_indices": list(self.matched_gold_indices),
            "matched_retrieved_indices": list(self.matched_retrieved_indices),
            "match_modes": list(self.match_modes),
            "recall": float(self.recall),
            "precision": float(self.precision),
            "full_support": float(self.full_support),
            "available": bool(self.available),
        }


@dataclass
class TrajectoryRewardResult:
    reward: TrajectoryReward
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def apply_to(self, trajectory: RolloutTrajectory) -> TrajectoryReward:
        trajectory.reward = self.reward
        trajectory.metadata["reward_diagnostics"] = self.diagnostics
        return self.reward


_SUPPORT_WHITESPACE_RE = re.compile(r"\s+")
_ANSWER_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_ANSWER_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_support_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    return _SUPPORT_WHITESPACE_RE.sub(" ", text)


def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(_ANSWER_PUNCT_TABLE)
    text = _ANSWER_ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def answer_exact_match(prediction: str, gold_answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def answer_token_f1(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return float(2.0 * precision * recall / (precision + recall))


def best_answer_scores(
    prediction: str,
    gold_answers: Sequence[str],
) -> Tuple[float, float]:
    nonempty_gold = [str(answer) for answer in gold_answers if str(answer).strip()]
    if not nonempty_gold:
        return 0.0, 0.0

    f1 = max(answer_token_f1(prediction, gold) for gold in nonempty_gold)
    em = max(answer_exact_match(prediction, gold) for gold in nonempty_gold)
    return float(f1), float(em)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def passage_title(passage: Any) -> str:
    metadata = _get(passage, "metadata", {}) or {}
    return _first_nonempty(
        (
            _get(passage, "title", None),
            _get(metadata, "title", None),
            _get(metadata, "paragraph_title", None),
            _get(metadata, "source_title", None),
        )
    )


def passage_text(passage: Any) -> str:
    metadata = _get(passage, "metadata", {}) or {}
    return _first_nonempty(
        (
            _get(passage, "text", None),
            _get(passage, "passage", None),
            _get(passage, "content", None),
            _get(metadata, "text", None),
            _get(metadata, "passage", None),
            _get(metadata, "content", None),
        )
    )


def passage_identifier(passage: Any) -> str:
    metadata = _get(passage, "metadata", {}) or {}
    return _first_nonempty(
        (
            _get(passage, "passage_id", None),
            _get(passage, "id", None),
            _get(passage, "idx", None),
            _get(metadata, "passage_id", None),
            _get(metadata, "source_id", None),
            _get(metadata, "idx", None),
            _get(metadata, "fallback_idx", None),
        )
    )


def _support_representation(passage: Any) -> Tuple[str, str]:
    return (
        normalize_support_text(passage_title(passage)),
        normalize_support_text(passage_text(passage)),
    )


def _match_mode(
    retrieved: Tuple[str, str],
    gold: Tuple[str, str],
    *,
    allow_title_only: bool,
    title_is_unique: bool,
) -> Optional[str]:
    retrieved_title, retrieved_text = retrieved
    gold_title, gold_text = gold

    if (
        retrieved_title
        and gold_title
        and retrieved_text
        and gold_text
        and retrieved_title == gold_title
        and retrieved_text == gold_text
    ):
        return "title_and_text"

    if retrieved_text and gold_text and retrieved_text == gold_text:
        return "text"

    if (
        allow_title_only
        and title_is_unique
        and retrieved_title
        and gold_title
        and retrieved_title == gold_title
    ):
        return "title"

    return None


def match_supporting_passages(
    retrieved_passages: Sequence[Any],
    gold_passages: Sequence[Any],
    *,
    top_k: int = 5,
    allow_title_only_match: bool = True,
    title_only_requires_unique: bool = True,
) -> SupportMatchResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    retrieved_top_k = list(retrieved_passages[:top_k])
    gold_list = list(gold_passages)

    result = SupportMatchResult(
        top_k=top_k,
        num_gold=len(gold_list),
        num_retrieved=len(retrieved_top_k),
        available=bool(gold_list),
    )
    if not gold_list:
        return result

    retrieved_repr = [_support_representation(p) for p in retrieved_top_k]
    gold_repr = [_support_representation(p) for p in gold_list]

    gold_title_counts = Counter(title for title, _ in gold_repr if title)
    retrieved_title_counts = Counter(title for title, _ in retrieved_repr if title)

    unmatched_gold = set(range(len(gold_list)))

    mode_priority = {"title_and_text": 3, "text": 2, "title": 1}

    for retrieved_index, retrieved_item in enumerate(retrieved_repr):
        candidates: List[Tuple[int, int, str]] = []
        retrieved_title = retrieved_item[0]

        for gold_index in unmatched_gold:
            gold_title = gold_repr[gold_index][0]
            title_is_unique = True
            if title_only_requires_unique:
                title_is_unique = bool(
                    gold_title
                    and gold_title_counts[gold_title] == 1
                    and retrieved_title_counts[retrieved_title] == 1
                )

            mode = _match_mode(
                retrieved_item,
                gold_repr[gold_index],
                allow_title_only=allow_title_only_match,
                title_is_unique=title_is_unique,
            )
            if mode is not None:
                candidates.append((mode_priority[mode], gold_index, mode))

        if not candidates:
            continue

        _, gold_index, mode = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        unmatched_gold.remove(gold_index)
        result.matched_gold_indices.append(gold_index)
        result.matched_retrieved_indices.append(retrieved_index)
        result.match_modes.append(mode)

    matched = len(result.matched_gold_indices)
    result.recall = matched / len(gold_list)
    result.precision = (
        matched / len(retrieved_top_k) if retrieved_top_k else 0.0
    )
    result.full_support = float(matched == len(gold_list))
    return result


def _canonical_list(values: Any) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        return ()

    normalized = {
        normalize_support_text(value)
        for value in values
        if normalize_support_text(value)
    }
    return tuple(sorted(normalized))


def _search_request_payload(action: RolloutAction) -> Mapping[str, Any]:
    parsed = action.parsed_action or {}
    if not isinstance(parsed, Mapping):
        return {}

    nested = parsed.get("search_request")
    if isinstance(nested, Mapping):
        return nested
    return parsed


def canonical_search_signature(action: RolloutAction) -> Tuple[Any, ...]:
    request = _search_request_payload(action)
    return (
        normalize_support_text(request.get("search_focus", "")),
        _canonical_list(request.get("seed_entities", [])),
        _canonical_list(request.get("relation_hints", [])),
    )


def duplicate_search_rate(actions: Sequence[RolloutAction]) -> Tuple[float, Dict[str, int]]:
    signatures: List[Tuple[Any, ...]] = []
    seen = set()
    duplicates = 0

    for action in actions:
        if action.action_name != "SearchGraph":
            continue
        signature = canonical_search_signature(action)
        signatures.append(signature)
        if signature in seen:
            duplicates += 1
        else:
            seen.add(signature)

    num_searches = len(signatures)
    denominator = max(num_searches - 1, 1)
    rate = duplicates / denominator if num_searches > 1 else 0.0
    return float(rate), {
        "num_search_actions": num_searches,
        "num_duplicate_searches": duplicates,
        "num_unique_searches": len(seen),
    }


def format_validity(actions: Sequence[RolloutAction]) -> Tuple[float, Dict[str, int]]:
    generated = [action for action in actions if action.generated_token_ids or action.generated_text]
    if not generated:
        return 0.0, {
            "num_generated_actions": 0,
            "num_valid_actions": 0,
            "num_fallback_actions": 0,
        }

    valid = sum(
        1
        for action in generated
        if action.parse_success and not action.fallback_used
    )
    fallbacks = sum(1 for action in generated if action.fallback_used)
    return valid / len(generated), {
        "num_generated_actions": len(generated),
        "num_valid_actions": valid,
        "num_fallback_actions": fallbacks,
    }


def _walk_for_passages(value: Any) -> Iterable[Any]:
    if value is None:
        return

    if isinstance(value, Mapping):
        # Direct SearchGraph result or nested AgentObservation.
        for key in ("passages", "retrieved_passages", "evidence_passages"):
            items = value.get(key)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                for item in items:
                    yield item

        for key in ("search_result", "observation", "result"):
            nested = value.get(key)
            if nested is not None:
                yield from _walk_for_passages(nested)
        return

    # Dataclass/object form.
    for key in ("passages", "retrieved_passages", "evidence_passages"):
        items = getattr(value, key, None)
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            for item in items:
                yield item

    for key in ("search_result", "observation", "result"):
        nested = getattr(value, key, None)
        if nested is not None:
            yield from _walk_for_passages(nested)


def _passage_content_key(passage: Any) -> str:
    identifier = passage_identifier(passage)
    if identifier:
        return f"id::{identifier}"

    title, text = _support_representation(passage)
    if title or text:
        return f"content::{title}||{text}"
    return ""


def evidence_novelty(actions: Sequence[RolloutAction]) -> Tuple[float, Dict[str, int]]:
    seen = set()
    total_returns = 0
    new_returns = 0
    observable_searches = 0

    for action in actions:
        if action.action_name != "SearchGraph":
            continue

        keys: List[str] = []
        local_seen = set()
        for passage in _walk_for_passages(action.observation_payload):
            key = _passage_content_key(passage)
            if not key or key in local_seen:
                continue
            local_seen.add(key)
            keys.append(key)

        if not keys:
            continue

        observable_searches += 1
        for key in keys:
            total_returns += 1
            if key not in seen:
                new_returns += 1
                seen.add(key)

    novelty = new_returns / total_returns if total_returns else 0.0
    return float(novelty), {
        "num_observable_searches": observable_searches,
        "num_observed_passage_returns": total_returns,
        "num_new_passage_returns": new_returns,
        "num_unique_observed_passages": len(seen),
    }


def is_unknown_answer(answer: str, unknown_answers: Sequence[str]) -> bool:
    normalized = normalize_answer(answer)
    if not normalized:
        return True

    normalized_unknowns = {normalize_answer(value) for value in unknown_answers}
    return normalized in normalized_unknowns


class TrajectoryRewardCalculator:
    def __init__(self, config: Optional[TrajectoryRewardConfig] = None):
        self.config = config or TrajectoryRewardConfig()

    def _final_evidence(self, trajectory: RolloutTrajectory) -> List[Any]:
        if trajectory.fused_evidence_passages:
            return list(trajectory.fused_evidence_passages)
        if trajectory.selected_evidence_passages:
            return list(trajectory.selected_evidence_passages)
        return list(trajectory.base_evidence_passages)

    def compute_result(self, trajectory: RolloutTrajectory) -> TrajectoryRewardResult:
        cfg = self.config
        prediction = trajectory.predicted_answer

        answer_f1_value, answer_em_value = best_answer_scores(
            prediction,
            trajectory.gold_answers,
        )
        answer_available = any(str(value).strip() for value in trajectory.gold_answers)

        final_evidence = self._final_evidence(trajectory)
        support_result = match_supporting_passages(
            retrieved_passages=final_evidence,
            gold_passages=trajectory.gold_supporting_passages,
            top_k=cfg.support_top_k,
            allow_title_only_match=cfg.allow_title_only_match,
            title_only_requires_unique=cfg.title_only_requires_unique,
        )

        format_value, format_stats = format_validity(trajectory.actions)
        novelty_value, novelty_stats = evidence_novelty(trajectory.actions)
        duplicate_value, duplicate_stats = duplicate_search_rate(trajectory.actions)

        search_calls = max(int(trajectory.num_search_calls), 0)
        search_cost = min(search_calls / cfg.max_search_calls, 1.0)
        forced_stop = float(
            trajectory.forced_stop
            or any(action.forced_stop for action in trajectory.actions)
        )
        unknown = float(is_unknown_answer(prediction, cfg.unknown_answers))

        k = cfg.support_top_k
        components = {
            "answer_f1": float(answer_f1_value),
            "answer_exact_match": float(answer_em_value),
            f"support_recall_at_{k}": float(support_result.recall),
            f"support_precision_at_{k}": float(support_result.precision),
            f"full_support_at_{k}": float(support_result.full_support),
            "format_validity": float(format_value),
            "evidence_novelty": float(novelty_value),
            "search_cost": float(search_cost),
            "duplicate_search_rate": float(duplicate_value),
            "forced_stop": float(forced_stop),
            "unknown_answer": float(unknown),
        }

        weights = cfg.component_weights()

        if not answer_available:
            weights["answer_f1"] = 0.0
            weights["answer_exact_match"] = 0.0
        if not support_result.available:
            weights[f"support_recall_at_{k}"] = 0.0
            weights[f"support_precision_at_{k}"] = 0.0
            weights[f"full_support_at_{k}"] = 0.0
        if novelty_stats["num_observed_passage_returns"] == 0:
            weights["evidence_novelty"] = 0.0

        reward = TrajectoryReward(
            components=components,
            weights=weights,
        )
        reward.recompute_total()
        reward.validate()

        diagnostics: Dict[str, Any] = {
            "prediction": prediction,
            "answer_supervision_available": answer_available,
            "support_supervision_available": support_result.available,
            "support_matching": support_result.to_dict(),
            "format": format_stats,
            "novelty": novelty_stats,
            "duplicate_search": duplicate_stats,
            "num_search_calls": search_calls,
            "max_search_calls": cfg.max_search_calls,
            "termination_reason": trajectory.termination_reason,
            "used_reader_answer": bool((trajectory.reader_predicted_answer or "").strip()),
            "effective_weights": dict(weights),
        }

        return TrajectoryRewardResult(reward=reward, diagnostics=diagnostics)

    def compute(
        self,
        trajectory: RolloutTrajectory,
        *,
        assign: bool = True,
    ) -> TrajectoryReward:
        result = self.compute_result(trajectory)
        if assign:
            result.apply_to(trajectory)
        return result.reward

    def score_group(
        self,
        group: RolloutGroup,
        *,
        compute_advantages: bool = True,
    ) -> List[float]:

        for trajectory in group.trajectories:
            self.compute(trajectory, assign=True)

        if not compute_advantages:
            return [float(t.reward.total) for t in group.trajectories]

        return group.compute_advantages(
            epsilon=self.config.advantage_epsilon,
            zero_variance_threshold=self.config.zero_variance_threshold,
        )


def compute_trajectory_reward(
    trajectory: RolloutTrajectory,
    config: Optional[TrajectoryRewardConfig] = None,
    *,
    assign: bool = True,
) -> TrajectoryReward:
    return TrajectoryRewardCalculator(config).compute(
        trajectory,
        assign=assign,
    )


def score_rollout_group(
    group: RolloutGroup,
    config: Optional[TrajectoryRewardConfig] = None,
    *,
    compute_advantages: bool = True,
) -> List[float]:

    return TrajectoryRewardCalculator(config).score_group(
        group,
        compute_advantages=compute_advantages,
    )