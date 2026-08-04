"""Important definition in this project:
    evidence = retrieved passages + filtered triples/facts + bridge entities
Reward components:
    answer correctness
    supporting passage recall
    full support recovery
    groundedness over passages + triples
    filtered triple usefulness
    bridge entity usage
    valid action/JSON format
    stop/search behavior
    repetition penalty
    search efficiency penalty
    retrieval noise penalty
GRPO later compares multiple sampled trajectories for the same question:
    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
"""

from __future__ import annotations
import ast
import json
import logging
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from .schema import AgentTrajectory, EvaluationResult, RetrievedPassage, RewardBreakdown

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:

    # Positive reward weights
    answer_weight: float = 0.35
    support_recall_weight: float = 0.25

    # Kept for logging/evaluation compatibility, but not used in R_total by default.
    full_support_weight: float = 0.0
    groundedness_weight: float = 0.0

    # Process rewards
    triple_reward_weight: float = 0.15
    bridge_reward_weight: float = 0.10
    format_reward_weight: float = 0.10
    stop_reward_weight: float = 0.05

    # Penalty weights
    step_penalty_weight: float = 0.03
    repeat_penalty_weight: float = 0.10

    # Kept for metadata/debugging, but not used in R_total by default.
    noise_penalty_weight: float = 0.0
    token_penalty_weight: float = 0.0

    # Retrieval evaluation
    support_k: int = 5
    noise_k: int = 5

    # Answer scoring
    use_max_over_gold: bool = True

    # Groundedness heuristic
    groundedness_answer_substring: bool = True

    # Safety
    unknown_answer_penalty: float = 0.0

    # Curriculum shaping
    # During early GRPO, efficiency penalties should be disabled or small.
    training_step: int = 0
    penalty_warmup_steps: int = 200
    penalty_ramp_steps: int = 300

    # Stop reward behavior
    min_support_for_stop: float = 0.5


_ARTICLES = {"a", "an", "the"}

def normalize_answer(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()

    text = "".join(ch for ch in text if ch not in string.punctuation)

    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in _ARTICLES]

    return " ".join(tokens)


def exact_match_score(prediction: Any, gold: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def max_answer_score(
    prediction: Any,
    gold_answers: Sequence[Any],
) -> Tuple[float, float]:
    if not gold_answers:
        return 0.0, 0.0

    em_scores = [exact_match_score(prediction, gold) for gold in gold_answers]
    f1_scores = [f1_score(prediction, gold) for gold in gold_answers]

    return max(em_scores), max(f1_scores)


def is_unknown_answer(answer: Any) -> bool:
    norm = normalize_answer(answer)
    return norm in {
        "",
        "i dont know",
        "unknown",
        "not enough information",
        "cannot determine",
        "none",
    }


def answer_in_text(
    gold_answers: Sequence[Any],
    text: str,
) -> float:
    text_norm = normalize_answer(text)

    if not text_norm:
        return 0.0

    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in text_norm:
            return 1.0

    return 0.0


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()

    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)

    return result


def support_recall_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    gold_set = set(gold_ids)

    if not gold_set:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & gold_set) / len(gold_set)


def support_precision_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    retrieved_at_k = list(retrieved_ids[:k])

    if not retrieved_at_k:
        return 0.0

    gold_set = set(gold_ids)
    return len(set(retrieved_at_k) & gold_set) / len(retrieved_at_k)


def full_support_recall_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    gold_set = set(gold_ids)

    if not gold_set:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    return float(gold_set.issubset(retrieved_set))


def noise_rate_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    retrieved_at_k = list(retrieved_ids[:k])

    if not retrieved_at_k:
        return 0.0

    gold_set = set(gold_ids)
    non_gold = [pid for pid in retrieved_at_k if pid not in gold_set]

    return len(non_gold) / len(retrieved_at_k)


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k_list: Sequence[int] = (1, 5, 10),
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    recall_at_k: Dict[int, float] = {}
    precision_at_k: Dict[int, float] = {}
    full_recall_at_k: Dict[int, float] = {}

    for k in k_list:
        recall_at_k[k] = support_recall_at_k(retrieved_ids, gold_ids, k)
        precision_at_k[k] = support_precision_at_k(retrieved_ids, gold_ids, k)
        full_recall_at_k[k] = full_support_recall_at_k(retrieved_ids, gold_ids, k)

    return recall_at_k, precision_at_k, full_recall_at_k


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_plain(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if hasattr(obj, "__dict__"):
        return {k: to_plain(v) for k, v in vars(obj).items()}

    return str(obj)


def parse_json_like_output(text: Any) -> Dict[str, Any]:
    if text is None:
        return {}

    text = str(text).strip()

    text = re.sub(r"^```(?:json|python)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            try:
                value = ast.literal_eval(candidate)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass

    try:
        value = ast.literal_eval(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    return {}


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def get_retrieved_passages_from_trajectory(
    trajectory: AgentTrajectory,
) -> List[RetrievedPassage]:
    passages: List[RetrievedPassage] = []
    seen = set()

    # Main source: step observations.
    for step in trajectory.steps:
        obs = step.observation
        if obs is None or obs.search_result is None:
            continue

        for passage in obs.search_result.passages:
            pid = passage.passage_id
            if pid in seen:
                continue

            passages.append(passage)
            seen.add(pid)

    return passages

def passage_aliases_from_passage(passage: Any) -> List[str]:
    aliases: List[str] = []

    for key in ["passage_id", "id", "idx", "chunk_id"]:
        value = safe_get(passage, key, None)
        if value is not None:
            aliases.append(str(value))

    metadata = safe_get(passage, "metadata", {}) or {}

    if isinstance(metadata, dict):
        for key in [
            "source_id",
            "idx",
            "original_idx",
            "original_id",
            "original_passage_id",
            "fallback_idx",
        ]:
            value = metadata.get(key)
            if value is not None:
                aliases.append(str(value))

    return unique_preserve_order(aliases)


def get_retrieved_alias_groups_from_trajectory(
    trajectory: AgentTrajectory,
) -> List[List[str]]:
    passages = get_retrieved_passages_from_trajectory(trajectory)
    return [passage_aliases_from_passage(p) for p in passages]


def support_recall_at_k_alias(
    retrieved_alias_groups: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    gold_set = set(str(x) for x in gold_ids)

    if not gold_set:
        return 0.0

    retrieved_aliases = set()

    for aliases in retrieved_alias_groups[:k]:
        retrieved_aliases.update(str(x) for x in aliases)

    return len(retrieved_aliases & gold_set) / len(gold_set)


def support_precision_at_k_alias(
    retrieved_alias_groups: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    top_groups = list(retrieved_alias_groups[:k])

    if not top_groups:
        return 0.0

    gold_set = set(str(x) for x in gold_ids)

    matched = 0
    for aliases in top_groups:
        if set(str(x) for x in aliases) & gold_set:
            matched += 1

    return matched / len(top_groups)


def full_support_recall_at_k_alias(
    retrieved_alias_groups: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    gold_set = set(str(x) for x in gold_ids)

    if not gold_set:
        return 0.0

    retrieved_aliases = set()

    for aliases in retrieved_alias_groups[:k]:
        retrieved_aliases.update(str(x) for x in aliases)

    return float(gold_set.issubset(retrieved_aliases))


def noise_rate_at_k_alias(
    retrieved_alias_groups: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    top_groups = list(retrieved_alias_groups[:k])

    if not top_groups:
        return 0.0

    gold_set = set(str(x) for x in gold_ids)

    noisy = 0
    for aliases in top_groups:
        if not (set(str(x) for x in aliases) & gold_set):
            noisy += 1

    return noisy / len(top_groups)


def get_filtered_triples_from_trajectory(
    trajectory: AgentTrajectory,
) -> List[Any]:
    triples: List[Any] = []
    seen = set()

    for step in trajectory.steps:
        obs = step.observation
        if obs is None or obs.search_result is None:
            continue

        filtered = getattr(obs.search_result, "filtered_triples", []) or []

        for triple in filtered:
            plain = to_plain(triple)
            key = json.dumps(plain, ensure_ascii=False, sort_keys=True)

            if key in seen:
                continue

            triples.append(triple)
            seen.add(key)

    return triples


def get_candidate_triples_from_trajectory(
    trajectory: AgentTrajectory,
) -> List[Any]:
    triples: List[Any] = []
    seen = set()

    for step in trajectory.steps:
        obs = step.observation
        if obs is None or obs.search_result is None:
            continue

        candidates = getattr(obs.search_result, "candidate_triples", []) or []

        for triple in candidates:
            plain = to_plain(triple)
            key = json.dumps(plain, ensure_ascii=False, sort_keys=True)

            if key in seen:
                continue

            triples.append(triple)
            seen.add(key)

    return triples


def triple_to_text(triple_obj: Any) -> str:
    plain = to_plain(triple_obj)

    if isinstance(plain, dict):
        triple = plain.get("triple", plain)

        if isinstance(triple, dict):
            subj = (
                triple.get("subject")
                or triple.get("subj")
                or triple.get("head")
                or ""
            )
            pred = (
                triple.get("predicate")
                or triple.get("relation")
                or triple.get("pred")
                or ""
            )
            obj = (
                triple.get("object")
                or triple.get("obj")
                or triple.get("tail")
                or ""
            )

            return f"{subj} {pred} {obj}".strip()

        return json.dumps(plain, ensure_ascii=False)

    if isinstance(plain, list):
        return " ".join(str(x) for x in plain)

    return str(plain)


def get_retrieved_ids_from_trajectory(
    trajectory: AgentTrajectory,
) -> List[str]:
    return [p.passage_id for p in get_retrieved_passages_from_trajectory(trajectory)]


def get_evidence_text_from_trajectory(
    trajectory: AgentTrajectory,
    max_chars: Optional[int] = None,
    include_triples: bool = True,
) -> str:
    passages = get_retrieved_passages_from_trajectory(trajectory)
    passage_text = "\n\n".join(p.text for p in passages)

    triple_text = ""
    if include_triples:
        triples = get_filtered_triples_from_trajectory(trajectory)
        triple_text = "\n\n".join(triple_to_text(t) for t in triples)

    text = "\n\n".join(x for x in [passage_text, triple_text] if x)

    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]

    return text


def token_proxy_from_trajectory(trajectory: AgentTrajectory) -> int:
    total_chars = 0

    for step in trajectory.steps:
        if step.action.reasoning_summary:
            total_chars += len(step.action.reasoning_summary)
        if step.action.raw_output:
            total_chars += len(step.action.raw_output)
        if step.action.search_request is not None:
            total_chars += len(step.action.search_request.build_search_query())

    return max(1, total_chars // 4)


def groundedness_score(
    answer: Any,
    evidence_text: str,
    gold_answers: Optional[Sequence[Any]] = None,
) -> float:
    if is_unknown_answer(answer):
        return 0.0

    evidence_norm = normalize_answer(evidence_text)

    answer_norm = normalize_answer(answer)
    if answer_norm and answer_norm in evidence_norm:
        return 1.0

    for gold in gold_answers or []:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in evidence_norm:
            return 1.0

    return 0.0


def format_reward_from_trajectory(trajectory: AgentTrajectory) -> float:
    if not trajectory.steps:
        return 0.0

    scores: List[float] = []

    for step in trajectory.steps:
        action = step.action
        action_name = getattr(action, "action", None)

        base = 0.0

        if action_name == "SearchGraph":
            req = action.search_request
            if req is not None:
                base += 0.5

                if str(getattr(req, "search_focus", "") or "").strip():
                    base += 0.2

                seed_entities = getattr(req, "seed_entities", []) or []
                relation_hints = getattr(req, "relation_hints", []) or []

                if seed_entities:
                    base += 0.15
                if relation_hints:
                    base += 0.15

        elif action_name == "SubmitFinalAnswer":
            base += 0.6
            if str(getattr(action, "answer", "") or "").strip():
                base += 0.4

        raw = getattr(action, "raw_output", None)

        # Synthetic fallback steps may have raw_output=None. Do not fully punish.
        if raw is None:
            json_bonus = 0.7
        else:
            parsed = parse_json_like_output(raw)
            json_bonus = 1.0 if parsed else 0.0

        scores.append(0.7 * clip01(base) + 0.3 * json_bonus)

    return clip01(sum(scores) / len(scores))


def triple_reward_score(
    trajectory: AgentTrajectory,
    gold_answers: Sequence[Any],
) -> float:
    filtered_triples = get_filtered_triples_from_trajectory(trajectory)

    if not filtered_triples:
        return 0.0

    triple_texts = [triple_to_text(t) for t in filtered_triples]
    joined_triples = "\n".join(triple_texts)

    if answer_in_text(gold_answers, joined_triples) > 0.0:
        return 1.0

    question_tokens = set(normalize_answer(trajectory.question).split())
    question_tokens = {
        tok for tok in question_tokens
        if len(tok) > 2 and tok not in {"what", "when", "where", "who", "which"}
    }

    if not question_tokens:
        return 0.3

    overlap_scores = []

    for text in triple_texts:
        toks = set(normalize_answer(text).split())
        if not toks:
            overlap_scores.append(0.0)
            continue

        overlap = len(toks & question_tokens) / max(1, len(question_tokens))
        overlap_scores.append(min(1.0, overlap))

    return min(0.8, sum(overlap_scores) / len(overlap_scores) + 0.2)


def bridge_entity_reward_score(trajectory: AgentTrajectory) -> float:
    seen_evidence_text = ""
    bridge_hits = 0
    possible_later_searches = 0

    for step in trajectory.steps:
        action = step.action

        if action.action == "SearchGraph" and action.search_request is not None:
            seeds = action.search_request.seed_entities or []

            if seen_evidence_text:
                possible_later_searches += 1
                seen_norm = normalize_answer(seen_evidence_text)

                for seed in seeds:
                    seed_norm = normalize_answer(seed)
                    if seed_norm and seed_norm in seen_norm:
                        bridge_hits += 1
                        break

        obs = step.observation
        if obs is not None and obs.search_result is not None:
            for passage in obs.search_result.passages:
                seen_evidence_text += "\n" + passage.text

            for triple in getattr(obs.search_result, "filtered_triples", []) or []:
                seen_evidence_text += "\n" + triple_to_text(triple)

    if possible_later_searches <= 0:
        return 0.0

    return clip01(bridge_hits / possible_later_searches)


def repeat_search_penalty_score(trajectory: AgentTrajectory) -> float:
    keys: List[str] = []

    for step in trajectory.steps:
        action = step.action

        if action.action != "SearchGraph" or action.search_request is None:
            continue

        req = action.search_request

        focus = normalize_answer(req.search_focus)
        seeds = "|".join(sorted(normalize_answer(x) for x in (req.seed_entities or [])))
        hints = "|".join(sorted(normalize_answer(x) for x in (req.relation_hints or [])))

        keys.append(f"{focus}::{seeds}::{hints}")

    if len(keys) <= 1:
        return 0.0

    unique_count = len(set(keys))
    repeat_count = len(keys) - unique_count

    return clip01(repeat_count / max(1, len(keys) - 1))


def stop_reward_score(
    trajectory: AgentTrajectory,
    answer_reward: float,
    support_recall: float,
    grounded: float,
    config: RewardConfig,
) -> float:
    prediction = trajectory.final_answer or ""

    if answer_reward > 0.8:
        return 1.0

    if answer_reward > 0.0 and grounded > 0.0:
        return 0.8

    if is_unknown_answer(prediction):
        # If it found meaningful support but still says I don't know, bad stop.
        if support_recall >= config.min_support_for_stop or grounded > 0.0:
            return 0.0
        return 0.3

    if support_recall >= config.min_support_for_stop and grounded > 0.0:
        return 0.5

    return 0.0


def penalty_scale(config: RewardConfig) -> float:
    
    step = max(0, int(config.training_step))
    warmup = max(0, int(config.penalty_warmup_steps))
    ramp = max(1, int(config.penalty_ramp_steps))

    if step < warmup:
        return 0.0

    return clip01((step - warmup) / ramp)


# ---------------------------------------------------------------------
# Main reward computer
# ---------------------------------------------------------------------


class RewardComputer:
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute(
        self,
        trajectory: AgentTrajectory,
        prediction_override: Optional[str] = None,
        prediction_source: str = "controller",
    ) -> RewardBreakdown:
        controller_prediction = trajectory.final_answer or ""

        if prediction_override is not None:
            prediction = str(prediction_override)
        else:
            prediction = controller_prediction

        gold_answers = trajectory.gold_answers
        gold_passage_ids = unique_preserve_order(trajectory.gold_passage_ids)

        retrieved_ids = unique_preserve_order(
            get_retrieved_ids_from_trajectory(trajectory)
        )
        retrieved_alias_groups = get_retrieved_alias_groups_from_trajectory(trajectory)

        em, f1 = max_answer_score(prediction, gold_answers)

        # Main answer reward now comes from 70B answer if prediction_override is provided.
        answer_reward = f1

        if is_unknown_answer(prediction):
            answer_reward = min(answer_reward, self.config.unknown_answer_penalty)

        support_recall = support_recall_at_k_alias(
            retrieved_alias_groups=retrieved_alias_groups,
            gold_ids=gold_passage_ids,
            k=self.config.support_k,
        )

        full_support = full_support_recall_at_k_alias(
            retrieved_alias_groups=retrieved_alias_groups,
            gold_ids=gold_passage_ids,
            k=self.config.support_k,
        )

        evidence_text = get_evidence_text_from_trajectory(
            trajectory,
            include_triples=True,
        )

        evidence_answer_hit = answer_in_text(
            gold_answers=gold_answers,
            text=evidence_text,
        )

        grounded = groundedness_score(
            answer=prediction,
            evidence_text=evidence_text,
            gold_answers=gold_answers,
        )

        triple_reward = triple_reward_score(
            trajectory=trajectory,
            gold_answers=gold_answers,
        )

        bridge_reward = bridge_entity_reward_score(trajectory)

        format_reward = format_reward_from_trajectory(trajectory)

        stop_reward = stop_reward_score(
            trajectory=trajectory,
            answer_reward=answer_reward,
            support_recall=support_recall,
            grounded=grounded,
            config=self.config,
        )

        num_search_calls = trajectory.num_search_calls()

        noise_rate = noise_rate_at_k_alias(
            retrieved_alias_groups=retrieved_alias_groups,
            gold_ids=gold_passage_ids,
            k=self.config.noise_k,
        )

        token_proxy = token_proxy_from_trajectory(trajectory)

        repeat_penalty_raw = repeat_search_penalty_score(trajectory)

        scale = penalty_scale(self.config)

        step_penalty = (
            scale
            * self.config.step_penalty_weight
            * float(num_search_calls)
        )

        repeat_penalty = (
            scale
            * self.config.repeat_penalty_weight
            * float(repeat_penalty_raw)
        )

        # Kept for logging only. Not included in R_total equation.
        noise_penalty = (
            scale
            * self.config.noise_penalty_weight
            * float(noise_rate)
        )

        token_penalty = (
            scale
            * self.config.token_penalty_weight
            * float(token_proxy)
        )

        positive_reward = (
            self.config.answer_weight * answer_reward
            + self.config.support_recall_weight * support_recall
            + self.config.triple_reward_weight * triple_reward
            + self.config.bridge_reward_weight * bridge_reward
            + self.config.format_reward_weight * format_reward
            + self.config.stop_reward_weight * stop_reward
        )

        raw_total_reward = (
            positive_reward
            - step_penalty
            - repeat_penalty
        )

        total_reward = clip01(raw_total_reward)

        return RewardBreakdown(
            total_reward=float(total_reward),
            answer_reward=float(answer_reward),
            support_recall_reward=float(support_recall),
            full_support_reward=float(full_support),
            groundedness_reward=float(grounded),
            step_penalty=float(step_penalty),
            token_penalty=float(token_penalty),
            noise_penalty=float(noise_penalty),
            metadata={
                "raw_total_reward": float(raw_total_reward),
                "positive_reward": float(positive_reward),

                "prediction": prediction,
                "prediction_source": prediction_source,
                "controller_prediction": controller_prediction,

                "exact_match": float(em),
                "f1": float(f1),

                "num_search_calls": int(num_search_calls),

                "retrieved_ids": retrieved_ids,
                "retrieved_alias_groups": retrieved_alias_groups,
                "gold_passage_ids": gold_passage_ids,

                "support_recall": float(support_recall),
                "full_support": float(full_support),

                "evidence_answer_hit": float(evidence_answer_hit),
                "groundedness": float(grounded),

                "noise_rate": float(noise_rate),
                "noise_penalty": float(noise_penalty),

                "repeat_penalty_raw": float(repeat_penalty_raw),
                "repeat_penalty": float(repeat_penalty),

                "token_proxy": int(token_proxy),
                "token_penalty": float(token_penalty),

                "num_filtered_triples": len(get_filtered_triples_from_trajectory(trajectory)),
                "num_candidate_triples": len(get_candidate_triples_from_trajectory(trajectory)),

                "triple_reward": float(triple_reward),
                "bridge_reward": float(bridge_reward),
                "format_reward": float(format_reward),
                "stop_reward": float(stop_reward),

                "penalty_scale": float(scale),
                "training_step": int(self.config.training_step),

                "reward_equation": (
                    "0.35*R_answer + 0.25*R_support + 0.15*R_triple "
                    "+ 0.10*R_bridge + 0.10*R_format + 0.05*R_stop "
                    "- lambda_repeat*P_repeat - lambda_step*P_step"
                ),
            },
        )

def evaluate_trajectory(
    trajectory: AgentTrajectory,
    k_list: Sequence[int] = (1, 5, 10),
) -> EvaluationResult:
    predicted_answer = trajectory.final_answer
    gold_answers = trajectory.gold_answers
    retrieved_ids = unique_preserve_order(get_retrieved_ids_from_trajectory(trajectory))
    gold_ids = unique_preserve_order(trajectory.gold_passage_ids)
    retrieved_alias_groups = get_retrieved_alias_groups_from_trajectory(trajectory)

    recall_at_k: Dict[int, float] = {}
    precision_at_k: Dict[int, float] = {}
    full_recall_at_k: Dict[int, float] = {}

    for k in k_list:
        recall_at_k[k] = support_recall_at_k_alias(
            retrieved_alias_groups=retrieved_alias_groups,
            gold_ids=gold_ids,
            k=k,
        )
        precision_at_k[k] = support_precision_at_k_alias(
            retrieved_alias_groups=retrieved_alias_groups,
            gold_ids=gold_ids,
            k=k,
        )
        full_recall_at_k[k] = full_support_recall_at_k_alias(
            retrieved_alias_groups=retrieved_alias_groups,
            gold_ids=gold_ids,
            k=k,
        )

    em, f1 = max_answer_score(predicted_answer or "", gold_answers)

    return EvaluationResult(
        question_id=trajectory.question_id,
        question=trajectory.question,
        predicted_answer=predicted_answer,
        gold_answers=list(gold_answers),
        retrieved_passage_ids=retrieved_ids,
        gold_passage_ids=gold_ids,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        full_recall_at_k=full_recall_at_k,
        exact_match=float(em),
        f1=float(f1),
        num_search_calls=trajectory.num_search_calls(),
        metadata={
            "num_steps": len(trajectory.steps),
            "stopped": trajectory.stopped(),
            "num_filtered_triples": len(get_filtered_triples_from_trajectory(trajectory)),
            "retrieved_alias_groups": retrieved_alias_groups,
        },
    )


def aggregate_evaluation_results(
    results: Sequence[EvaluationResult],
    k_list: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    if not results:
        return {
            "num_examples": 0,
        }

    summary: Dict[str, Any] = {
        "num_examples": len(results),
        "exact_match": _mean([r.exact_match or 0.0 for r in results]),
        "f1": _mean([r.f1 or 0.0 for r in results]),
        "num_search_calls": _mean([r.num_search_calls or 0.0 for r in results]),
    }

    for k in k_list:
        summary[f"recall@{k}"] = _mean([r.recall_at_k.get(k, 0.0) for r in results])
        summary[f"precision@{k}"] = _mean([r.precision_at_k.get(k, 0.0) for r in results])
        summary[f"full_recall@{k}"] = _mean(
            [r.full_recall_at_k.get(k, 0.0) for r in results]
        )

    return summary


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_group_advantages(
    rewards: Sequence[float],
    eps: float = 1e-8,
) -> List[float]:
    rewards = [float(r) for r in rewards]

    if not rewards:
        return []

    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = math.sqrt(variance)

    if std_reward < eps:
        return [0.0 for _ in rewards]

    return [(r - mean_reward) / (std_reward + eps) for r in rewards]


def score_trajectory(
    trajectory: AgentTrajectory,
    config: Optional[RewardConfig] = None,
    prediction_override: Optional[str] = None,
    prediction_source: str = "controller",
) -> RewardBreakdown:
    return RewardComputer(config).compute(
        trajectory,
        prediction_override=prediction_override,
        prediction_source=prediction_source,
    )