from __future__ import annotations
import argparse
import json
import os
import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "examples", "results", "answers", "trajectories"]:
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unsupported JSON format: {path}")


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def text_contains_any_answer(text: str, gold_answers: Sequence[str]) -> float:
    text_norm = normalize_answer(text)

    if not text_norm:
        return 0.0

    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in text_norm:
            return 1.0

    return 0.0


def unique_preserve_order(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()

    for value in values:
        if value is None:
            continue

        item = str(value)

        if not item or item in seen:
            continue

        output.append(item)
        seen.add(item)

    return output


def passage_aliases(passage: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []

    for key in ["passage_id", "id", "idx", "chunk_id"]:
        value = passage.get(key)
        if value is not None:
            aliases.append(str(value))

    metadata = passage.get("metadata")
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


def normalize_match_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def passage_title(passage: Dict[str, Any]) -> str:
    title = passage.get("title")
    if not title:
        metadata = passage.get("metadata")
        if isinstance(metadata, dict):
            title = metadata.get("title")
    return normalize_match_text(title)


def passage_body(passage: Dict[str, Any]) -> str:
    value = (
        passage.get("text")
        or passage.get("content")
        or passage.get("passage")
        or ""
    )
    return normalize_match_text(value)


def get_gold_support_passages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    paragraphs = (
        row.get("paragraphs")
        or row.get("contexts")
        or row.get("context")
        or []
    )
    if not isinstance(paragraphs, list):
        return []

    output: List[Dict[str, Any]] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            continue
        is_supporting = (
            paragraph.get("is_supporting")
            if "is_supporting" in paragraph
            else paragraph.get("supporting")
        )
        if bool(is_supporting):
            output.append(paragraph)
    return output


def count_gold_matches(
    passages: Sequence[Dict[str, Any]],
    gold_passages: Sequence[Dict[str, Any]],
    k: int,
) -> int:
    top_passages = list(passages[:k])
    unmatched_gold = set(range(len(gold_passages)))
    matched = 0

    for passage in top_passages:
        p_title = passage_title(passage)
        p_text = passage_body(passage)
        best_index = None
        best_priority = 99

        for gold_index in unmatched_gold:
            gold = gold_passages[gold_index]
            g_title = passage_title(gold)
            g_text = passage_body(gold)
            priority = 99

            if (
                p_title and g_title and p_text and g_text
                and p_title == g_title and p_text == g_text
            ):
                priority = 1
            elif p_text and g_text and p_text == g_text:
                priority = 2
            elif p_title and g_title and p_title == g_title:
                priority = 3

            if priority < best_priority:
                best_priority = priority
                best_index = gold_index

        if best_index is not None:
            unmatched_gold.remove(best_index)
            matched += 1

    return matched


def recall_at_k_from_gold_passages(
    passages: Sequence[Dict[str, Any]],
    gold_passages: Sequence[Dict[str, Any]],
    k: int,
) -> float:
    if not gold_passages:
        return 0.0
    return count_gold_matches(passages, gold_passages, k) / len(gold_passages)


def precision_at_k_from_gold_passages(
    passages: Sequence[Dict[str, Any]],
    gold_passages: Sequence[Dict[str, Any]],
    k: int,
) -> float:
    top_passages = list(passages[:k])
    if not top_passages:
        return 0.0
    return count_gold_matches(passages, gold_passages, k) / len(top_passages)


def full_support_at_k_from_gold_passages(
    passages: Sequence[Dict[str, Any]],
    gold_passages: Sequence[Dict[str, Any]],
    k: int,
) -> float:
    if not gold_passages:
        return 0.0
    return float(count_gold_matches(passages, gold_passages, k) == len(gold_passages))


def recall_at_k_from_passages(
    passages: Sequence[Dict[str, Any]],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    gold_set = set(str(x) for x in gold_ids)

    if not gold_set:
        return 0.0

    retrieved_aliases = set()

    for passage in passages[:k]:
        retrieved_aliases.update(passage_aliases(passage))

    return len(retrieved_aliases & gold_set) / len(gold_set)


def precision_at_k_from_passages(
    passages: Sequence[Dict[str, Any]],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    top_passages = list(passages[:k])

    if not top_passages:
        return 0.0

    gold_set = set(str(x) for x in gold_ids)

    matched_passages = 0
    for passage in top_passages:
        aliases = set(passage_aliases(passage))
        if aliases & gold_set:
            matched_passages += 1

    return matched_passages / len(top_passages)


def full_support_at_k_from_passages(
    passages: Sequence[Dict[str, Any]],
    gold_ids: Sequence[str],
    k: int,
) -> float:
    gold_set = set(str(x) for x in gold_ids)

    if not gold_set:
        return 0.0

    retrieved_aliases = set()

    for passage in passages[:k]:
        retrieved_aliases.update(passage_aliases(passage))

    return float(gold_set.issubset(retrieved_aliases))


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def get_question_id(row: Dict[str, Any], fallback: int = 0) -> str:
    return str(
        row.get("question_id")
        or row.get("qid")
        or row.get("id")
        or fallback
    )


def index_by_question_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}

    for i, row in enumerate(rows):
        qid = get_question_id(row, i)
        output[qid] = row

    return output


def attach_question_rows(
    rows: Sequence[Dict[str, Any]],
    question_rows: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if question_rows is None:
        return [dict(row) for row in rows]

    question_by_qid = index_by_question_id(question_rows)
    output: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        merged = dict(row)
        qid = get_question_id(row, index)
        source = question_by_qid.get(qid)
        if source is None and index < len(question_rows):
            source = question_rows[index]

        if isinstance(source, dict):
            if not merged.get("paragraphs") and source.get("paragraphs"):
                merged["paragraphs"] = source["paragraphs"]
            if not merged.get("gold_answers"):
                answers = source.get("answer") or source.get("answers")
                if answers is not None:
                    merged["gold_answers"] = as_list(answers)
            merged["_question_source_attached"] = True
        else:
            merged["_question_source_attached"] = False
        output.append(merged)

    return output


def merge_answer_and_retrieval_rows(
    answer_rows: Sequence[Dict[str, Any]],
    retrieval_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if retrieval_rows is None:
        return [dict(row) for row in answer_rows]

    retrieval_by_qid = index_by_question_id(retrieval_rows)

    merged: List[Dict[str, Any]] = []

    for i, answer_row in enumerate(answer_rows):
        qid = get_question_id(answer_row, i)
        retrieval_row = retrieval_by_qid.get(qid, {})

        row = dict(retrieval_row)
        row.update(answer_row)

        if "steps" not in row and "steps" in retrieval_row:
            row["steps"] = retrieval_row["steps"]

        if "retrieval_metadata" not in row:
            row["retrieval_metadata"] = retrieval_row.get("metadata", {})

        if "controller_final_answer" not in row:
            row["controller_final_answer"] = retrieval_row.get("final_answer")

        if not row.get("evidence_passages") and retrieval_row.get("evidence_passages"):
            row["evidence_passages"] = retrieval_row["evidence_passages"]

        if not row.get("filtered_triples") and retrieval_row.get("filtered_triples"):
            row["filtered_triples"] = retrieval_row["filtered_triples"]

        merged.append(row)

    return merged


def get_question(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or "")


def get_prediction(row: Dict[str, Any]) -> str:
    return str(
        row.get("predicted_answer")
        or row.get("prediction")
        or row.get("answer_prediction")
        or row.get("answer")
        or ""
    )


def get_gold_answers(row: Dict[str, Any]) -> List[str]:
    value = (
        row.get("gold_answers")
        or row.get("answers")
        or row.get("gold")
        or []
    )

    return [str(x) for x in as_list(value)]


def get_gold_passage_ids(row: Dict[str, Any]) -> List[str]:
    value = (
        row.get("gold_passage_ids")
        or row.get("gold_support_ids")
        or row.get("supporting_passage_ids_gold")
        or []
    )

    ids: List[str] = []

    for item in as_list(value):
        if isinstance(item, dict):
            pid = item.get("passage_id") or item.get("id") or item.get("idx")
            if pid is not None:
                ids.append(str(pid))
        elif isinstance(item, (list, tuple)) and item:
            ids.append(str(item[0]))
        else:
            ids.append(str(item))

    return unique_preserve_order(ids)


def get_named_passages(
    row: Dict[str, Any],
    field_name: str,
) -> List[Dict[str, Any]]:
    passages = row.get(field_name)

    if not isinstance(passages, list):
        return []

    return [
        passage
        for passage in passages
        if isinstance(passage, dict)
    ]


def get_evidence_passages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    passages = get_named_passages(
        row,
        "evidence_passages",
    )

    if passages:
        return passages

    passages = get_named_passages(
        row,
        "fused_evidence_passages",
    )

    if passages:
        return passages

    output: List[Dict[str, Any]] = []

    steps = row.get("steps") or []
    if not isinstance(steps, list):
        return output

    for step in steps:
        if not isinstance(step, dict):
            continue

        obs = step.get("observation") or {}
        if not isinstance(obs, dict):
            continue

        sr = obs.get("search_result") or {}
        if not isinstance(sr, dict):
            continue

        step_passages = sr.get("passages") or []
        if not isinstance(step_passages, list):
            continue

        for p in step_passages:
            if isinstance(p, dict):
                output.append(p)

    return output



def get_retrieved_passage_ids(row: Dict[str, Any]) -> List[str]:
    passages = get_evidence_passages(row)

    ids: List[str] = []
    for p in passages:
        pid = (
            p.get("passage_id")
            or p.get("id")
            or p.get("idx")
            or p.get("chunk_id")
        )

        if pid is not None:
            ids.append(str(pid))

    return unique_preserve_order(ids)


def get_passage_text(row: Dict[str, Any]) -> str:
    passages = get_evidence_passages(row)

    texts: List[str] = []
    for p in passages:
        text = p.get("text") or p.get("content") or p.get("passage") or ""
        if text:
            texts.append(str(text))

    return "\n\n".join(texts)


def triple_to_text(triple_obj: Any) -> str:
    if isinstance(triple_obj, dict):
        triple = triple_obj.get("triple", triple_obj)

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

        return json.dumps(triple_obj, ensure_ascii=False)

    if isinstance(triple_obj, list):
        return " ".join(str(x) for x in triple_obj)

    return str(triple_obj)


def get_filtered_triples(row: Dict[str, Any]) -> List[Any]:
    triples = row.get("filtered_triples")

    if isinstance(triples, list) and triples:
        return triples

    output: List[Any] = []

    steps = row.get("steps") or []
    if not isinstance(steps, list):
        return output

    for step in steps:
        if not isinstance(step, dict):
            continue

        obs = step.get("observation") or {}
        if not isinstance(obs, dict):
            continue

        sr = obs.get("search_result") or {}
        if not isinstance(sr, dict):
            continue

        step_triples = sr.get("filtered_triples") or []
        if isinstance(step_triples, list):
            output.extend(step_triples)

    return output


def get_filtered_triple_text(row: Dict[str, Any]) -> str:
    triples = get_filtered_triples(row)
    return "\n".join(triple_to_text(t) for t in triples)


def get_num_search_calls(row: Dict[str, Any]) -> int:
    # Prefer retrieval metadata.
    for key in ["metadata", "retrieval_metadata"]:
        metadata = row.get(key)
        if isinstance(metadata, dict):
            if metadata.get("num_search_calls") is not None:
                try:
                    return int(metadata["num_search_calls"])
                except Exception:
                    pass

    steps = row.get("steps") or []
    if not isinstance(steps, list):
        return 0

    count = 0
    for step in steps:
        if not isinstance(step, dict):
            continue

        action = step.get("action") or {}
        if not isinstance(action, dict):
            continue

        if action.get("action") == "SearchGraph":
            count += 1

    return count


def get_num_steps(row: Dict[str, Any]) -> int:
    steps = row.get("steps")
    if isinstance(steps, list):
        return len(steps)
    return 0


def get_controller_answer(row: Dict[str, Any]) -> str:
    return str(row.get("controller_final_answer") or row.get("final_answer") or "")


def get_finalization_metadata(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = row.get("metadata")

    if not isinstance(metadata, dict):
        return {}

    finalization = metadata.get("finalization")

    if not isinstance(finalization, dict):
        return {}

    return finalization


def get_selector_metadata(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    selector = get_finalization_metadata(row).get("selector")
    return selector if isinstance(selector, dict) else {}


def get_fusion_metadata(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    fusion = get_finalization_metadata(row).get("fusion")
    return fusion if isinstance(fusion, dict) else {}


def get_answer_reader_metadata(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    reader = get_finalization_metadata(row).get("answer_reader")
    return reader if isinstance(reader, dict) else {}


def get_invalid_support_id_count(
    row: Dict[str, Any],
) -> int:
    values = row.get("invalid_supporting_passage_ids")

    if isinstance(values, list):
        return len(values)

    reader = get_answer_reader_metadata(row)
    value = reader.get("num_invalid_support_ids")

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def has_matching_gold_support_id_space(
    passages: Sequence[Dict[str, Any]],
    gold_ids: Sequence[str],
) -> bool:
    if not passages or not gold_ids:
        return False

    gold_set = set(str(x) for x in gold_ids)
    aliases = set()

    for passage in passages:
        aliases.update(passage_aliases(passage))

    return bool(aliases & gold_set)


def evaluate_one(
    row: Dict[str, Any],
    index: int,
    k_list: Sequence[int],
) -> Dict[str, Any]:
    qid = get_question_id(row, index)
    question = get_question(row)

    prediction = get_prediction(row)
    gold_answers = get_gold_answers(row)
    evidence_passages = get_evidence_passages(row)
    base_passages = get_named_passages(
        row,
        "base_evidence_passages",
    )
    selector_passages = get_named_passages(
        row,
        "llm_selected_passages",
    )
    fused_passages = get_named_passages(
        row,
        "fused_evidence_passages",
    )

    if not fused_passages:
        fused_passages = evidence_passages

    retrieved_ids = get_retrieved_passage_ids(row)
    gold_passage_ids = get_gold_passage_ids(row)
    gold_support_passages = get_gold_support_passages(row)
    use_content_matching = bool(gold_support_passages)

    em, f1 = max_answer_score(
        prediction,
        gold_answers,
    )

    passage_text = get_passage_text(row)
    triple_text = get_filtered_triple_text(row)

    evidence_text = "\n\n".join(
        item
        for item in [
            passage_text,
            triple_text,
        ]
        if item
    )

    passage_answer_hit = text_contains_any_answer(
        passage_text,
        gold_answers,
    )
    filtered_triple_answer_hit = text_contains_any_answer(
        triple_text,
        gold_answers,
    )
    evidence_answer_hit = text_contains_any_answer(
        evidence_text,
        gold_answers,
    )

    controller_answer = get_controller_answer(row)

    selector_metadata = get_selector_metadata(row)
    fusion_metadata = get_fusion_metadata(row)
    reader_metadata = get_answer_reader_metadata(row)

    selector_fallback = bool(
        selector_metadata.get("fallback_used", False)
    )
    selector_skipped = bool(
        selector_metadata.get("skipped", False)
    )

    reader_parse_success_value = reader_metadata.get(
        "parse_success"
    )

    if reader_parse_success_value is None:
        reader_parse_success_value = row.get(
            "parse_success"
        )

    reader_parse_success = bool(
        reader_parse_success_value
    )

    result: Dict[str, Any] = {
        "question_id": qid,
        "question": question,
        "predicted_answer": prediction,
        "gold_answers": gold_answers,
        "exact_match": float(em),
        "f1": float(f1),

        "controller_answer": controller_answer,
        "controller_unknown": is_unknown_answer(
            controller_answer
        ),
        "is_unknown_prediction": is_unknown_answer(
            prediction
        ),

        "retrieved_passage_ids": retrieved_ids,
        "gold_passage_ids": gold_passage_ids,
        "num_gold_support_passages": len(gold_support_passages),
        "retrieval_match_mode": "title_text" if use_content_matching else "id",
        "question_source_attached": bool(row.get("_question_source_attached", False)),
        "num_evidence_passages": len(evidence_passages),
        "num_base_evidence_passages": len(base_passages),
        "num_selector_passages": len(selector_passages),
        "num_fused_passages": len(fused_passages),
        "num_filtered_triples": len(
            get_filtered_triples(row)
        ),
        "num_search_calls": get_num_search_calls(row),
        "num_steps": get_num_steps(row),

        # Finalization health.
        "has_integrated_finalization": bool(
            get_finalization_metadata(row)
        ),
        "selector_fallback": selector_fallback,
        "selector_skipped": selector_skipped,
        "reader_parse_success": reader_parse_success,
        "num_invalid_support_ids": (
            get_invalid_support_id_count(row)
        ),

        "passage_answer_hit": float(
            passage_answer_hit
        ),
        "filtered_triple_answer_hit": float(
            filtered_triple_answer_hit
        ),
        "evidence_answer_hit": float(
            evidence_answer_hit
        ),
        "gold_id_overlap_exists": (
            has_matching_gold_support_id_space(
                evidence_passages,
                gold_passage_ids,
            )
        ),
    }

    for k in k_list:
        def stage_metrics(stage_passages: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
            if use_content_matching:
                return (
                    recall_at_k_from_gold_passages(stage_passages, gold_support_passages, k),
                    precision_at_k_from_gold_passages(stage_passages, gold_support_passages, k),
                    full_support_at_k_from_gold_passages(stage_passages, gold_support_passages, k),
                )
            return (
                recall_at_k_from_passages(stage_passages, gold_passage_ids, k),
                precision_at_k_from_passages(stage_passages, gold_passage_ids, k),
                full_support_at_k_from_passages(stage_passages, gold_passage_ids, k),
            )

        recall, precision, full_support = stage_metrics(evidence_passages)
        result[f"recall@{k}"] = recall
        result[f"precision@{k}"] = precision
        result[f"full_support@{k}"] = full_support

        for stage_name, stage_passages in [
            ("base", base_passages),
            ("selector", selector_passages),
            ("fused", fused_passages),
        ]:
            if not stage_passages:
                result[f"{stage_name}_recall@{k}"] = None
                result[f"{stage_name}_full_support@{k}"] = None
                continue
            stage_recall, _, stage_full = stage_metrics(stage_passages)
            result[f"{stage_name}_recall@{k}"] = stage_recall
            result[f"{stage_name}_full_support@{k}"] = stage_full

    return result

def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return float(sum(values) / len(values))


def mean_defined(values: Sequence[Optional[float]]) -> Optional[float]:
    defined = [
        float(value)
        for value in values
        if value is not None
    ]

    if not defined:
        return None

    return mean(defined)


def aggregate_results(
    per_example: Sequence[Dict[str, Any]],
    k_list: Sequence[int],
) -> Dict[str, Any]:
    if not per_example:
        return {"num_examples": 0}

    summary: Dict[str, Any] = {
        "num_examples": len(per_example),

        # Answer metrics.
        "exact_match": mean(
            [item["exact_match"] for item in per_example]
        ),
        "f1": mean(
            [item["f1"] for item in per_example]
        ),
        "unknown_prediction_rate": mean(
            [
                float(item["is_unknown_prediction"])
                for item in per_example
            ]
        ),

        "controller_unknown_rate": mean(
            [
                float(item["controller_unknown"])
                for item in per_example
            ]
        ),
        "avg_num_search_calls": mean(
            [
                float(item["num_search_calls"])
                for item in per_example
            ]
        ),
        "avg_num_steps": mean(
            [
                float(item["num_steps"])
                for item in per_example
            ]
        ),


        "avg_num_evidence_passages": mean(
            [
                float(item["num_evidence_passages"])
                for item in per_example
            ]
        ),
        "avg_num_base_evidence_passages": mean(
            [
                float(
                    item["num_base_evidence_passages"]
                )
                for item in per_example
            ]
        ),
        "avg_num_selector_passages": mean(
            [
                float(item["num_selector_passages"])
                for item in per_example
            ]
        ),
        "avg_num_fused_passages": mean(
            [
                float(item["num_fused_passages"])
                for item in per_example
            ]
        ),
        "avg_num_filtered_triples": mean(
            [
                float(item["num_filtered_triples"])
                for item in per_example
            ]
        ),

        "integrated_finalization_rate": mean(
            [
                float(item["has_integrated_finalization"])
                for item in per_example
            ]
        ),
        "selector_fallback_rate": mean(
            [
                float(item["selector_fallback"])
                for item in per_example
            ]
        ),
        "selector_skipped_rate": mean(
            [
                float(item["selector_skipped"])
                for item in per_example
            ]
        ),
        "reader_parse_success_rate": mean(
            [
                float(item["reader_parse_success"])
                for item in per_example
            ]
        ),
        "avg_num_invalid_support_ids": mean(
            [
                float(item["num_invalid_support_ids"])
                for item in per_example
            ]
        ),

        "passage_answer_hit_rate": mean(
            [
                float(item["passage_answer_hit"])
                for item in per_example
            ]
        ),
        "filtered_triple_answer_hit_rate": mean(
            [
                float(
                    item["filtered_triple_answer_hit"]
                )
                for item in per_example
            ]
        ),
        "evidence_answer_hit_rate": mean(
            [
                float(item["evidence_answer_hit"])
                for item in per_example
            ]
        ),

        "gold_id_overlap_rate": mean(
            [float(item["gold_id_overlap_exists"]) for item in per_example]
        ),
        "question_source_attachment_rate": mean(
            [float(item["question_source_attached"]) for item in per_example]
        ),
        "content_match_evaluation_rate": mean(
            [float(item["retrieval_match_mode"] == "title_text") for item in per_example]
        ),
    }

    examples_with_gold_support = [
        item
        for item in per_example
        if (
            len(item.get("gold_passage_ids", [])) > 0
            or item.get("num_gold_support_passages", 0) > 0
        )
    ]

    summary["num_examples_with_gold_support"] = len(
        examples_with_gold_support
    )

    for k in k_list:
        summary[f"recall@{k}"] = mean(
            [
                item[f"recall@{k}"]
                for item in examples_with_gold_support
            ]
        )
        summary[f"precision@{k}"] = mean(
            [
                item[f"precision@{k}"]
                for item in examples_with_gold_support
            ]
        )
        summary[f"full_support@{k}"] = mean(
            [
                item[f"full_support@{k}"]
                for item in examples_with_gold_support
            ]
        )

        for stage in [
            "base",
            "selector",
            "fused",
        ]:
            summary[f"{stage}_recall@{k}"] = mean_defined(
                [
                    item.get(f"{stage}_recall@{k}")
                    for item in examples_with_gold_support
                ]
            )
            summary[
                f"{stage}_full_support@{k}"
            ] = mean_defined(
                [
                    item.get(
                        f"{stage}_full_support@{k}"
                    )
                    for item in examples_with_gold_support
                ]
            )

    return summary

def parse_k_list(text: str) -> List[int]:
    values: List[int] = []

    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        values.append(int(part))

    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PPR-agent retrieval and answer outputs."
    )

    parser.add_argument(
        "--questions_path",
        type=str,
        default=None,
        help=(
            "Original MuSiQue JSON/JSONL containing paragraphs and is_supporting labels. "
            "Used to fix local-ID versus global chunk-ID mismatches by matching title/text."
        ),
    )
    parser.add_argument(
        "--integrated_path",
        type=str,
        default=None,
        help=(
            "Path to the integrated run_agent_retrieval.py JSONL containing "
            "predicted_answer and finalized evidence. Recommended for the "
            "new integrated pipeline."
        ),
    )
    parser.add_argument(
        "--answers_path",
        type=str,
        default=None,
        help=(
            "Path to standalone answer-generation JSONL from the old "
            "unintegrated pipeline."
        ),
    )
    parser.add_argument(
        "--retrieval_path",
        type=str,
        default=None,
        help="Optional raw retrieval trajectory JSONL. Used for steps/search-call metrics.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to summary JSON.",
    )
    parser.add_argument(
        "--per_example_output_path",
        type=str,
        default=None,
        help="Optional path to per-example JSONL.",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="1,5,10",
        help="Comma/space separated k values.",
    )
    parser.add_argument("--limit", type=int, default=None)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    provided_primary_inputs = [
        path
        for path in [
            args.integrated_path,
            args.answers_path,
        ]
        if path is not None
    ]

    if len(provided_primary_inputs) > 1:
        raise ValueError(
            "Use only one of --integrated_path or --answers_path."
        )

    if (
        args.integrated_path is None
        and args.answers_path is None
        and args.retrieval_path is None
    ):
        raise ValueError(
            "Provide --integrated_path, --answers_path, "
            "or --retrieval_path."
        )

    integrated_rows: Optional[List[Dict[str, Any]]] = None
    answer_rows: Optional[List[Dict[str, Any]]] = None
    retrieval_rows: Optional[List[Dict[str, Any]]] = None

    if args.integrated_path is not None:
        if not os.path.exists(args.integrated_path):
            raise FileNotFoundError(args.integrated_path)

        integrated_rows = load_json_or_jsonl(
            args.integrated_path
        )

    if args.answers_path is not None:
        if not os.path.exists(args.answers_path):
            raise FileNotFoundError(args.answers_path)

        answer_rows = load_json_or_jsonl(
            args.answers_path
        )

    if args.retrieval_path is not None:
        if not os.path.exists(args.retrieval_path):
            raise FileNotFoundError(args.retrieval_path)

        retrieval_rows = load_json_or_jsonl(
            args.retrieval_path
        )

    if integrated_rows is not None:
        rows = integrated_rows
        input_path = args.integrated_path

    elif answer_rows is not None:
        rows = merge_answer_and_retrieval_rows(
            answer_rows=answer_rows,
            retrieval_rows=retrieval_rows,
        )
        input_path = args.answers_path

    else:
        rows = retrieval_rows or []
        input_path = args.retrieval_path

    question_rows: Optional[List[Dict[str, Any]]] = None
    if args.questions_path is not None:
        if not os.path.exists(args.questions_path):
            raise FileNotFoundError(args.questions_path)
        question_rows = load_json_or_jsonl(args.questions_path)

    rows = attach_question_rows(rows, question_rows)

    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    k_list = parse_k_list(args.k_list)

    per_example = [
        evaluate_one(row=row, index=i, k_list=k_list)
        for i, row in enumerate(rows)
    ]

    summary = aggregate_results(per_example, k_list=k_list)
    summary["questions_path"] = args.questions_path
    summary["integrated_path"] = args.integrated_path
    summary["answers_path"] = args.answers_path
    summary["retrieval_path"] = args.retrieval_path
    summary["input_path"] = input_path
    summary["k_list"] = k_list

    if (
        summary.get("content_match_evaluation_rate", 0.0) < 1.0
        and summary.get("gold_id_overlap_rate", 0.0) == 0.0
    ):
        summary["warning"] = (
            "Some examples were evaluated using incompatible passage IDs. "
            "Provide --questions_path so retrieval metrics use gold paragraph title/text matching."
        )

    save_json(args.output_path, summary)

    if args.per_example_output_path:
        save_jsonl(args.per_example_output_path, per_example)

    print("\nEvaluation finished.")
    print(f"num_examples: {summary['num_examples']}")
    print(f"EM: {summary.get('exact_match', 0.0):.4f}")
    print(f"F1: {summary.get('f1', 0.0):.4f}")

    for k in k_list:
        print(f"Recall@{k}: {summary.get(f'recall@{k}', 0.0):.4f}")
        print(f"Precision@{k}: {summary.get(f'precision@{k}', 0.0):.4f}")
        print(f"FullSupport@{k}: {summary.get(f'full_support@{k}', 0.0):.4f}")

        for stage in ["base", "selector", "fused"]:
            stage_recall = summary.get(
                f"{stage}_recall@{k}"
            )
            stage_full = summary.get(
                f"{stage}_full_support@{k}"
            )

            if stage_recall is not None:
                print(
                    f"{stage.capitalize()} Recall@{k}: "
                    f"{stage_recall:.4f}"
                )

            if stage_full is not None:
                print(
                    f"{stage.capitalize()} FullSupport@{k}: "
                    f"{stage_full:.4f}"
                )

    print(f"Unknown prediction rate: {summary.get('unknown_prediction_rate', 0.0):.4f}")
    print(f"Controller unknown rate: {summary.get('controller_unknown_rate', 0.0):.4f}")
    print(f"Avg search calls: {summary.get('avg_num_search_calls', 0.0):.4f}")
    print(f"Avg steps: {summary.get('avg_num_steps', 0.0):.4f}")
    print(f"Avg evidence passages: {summary.get('avg_num_evidence_passages', 0.0):.4f}")
    print(f"Avg filtered triples: {summary.get('avg_num_filtered_triples', 0.0):.4f}")

    print(
        "Integrated finalization rate: "
        f"{summary.get('integrated_finalization_rate', 0.0):.4f}"
    )
    print(
        "Selector fallback rate: "
        f"{summary.get('selector_fallback_rate', 0.0):.4f}"
    )
    print(
        "Selector skipped rate: "
        f"{summary.get('selector_skipped_rate', 0.0):.4f}"
    )
    print(
        "Reader parse success rate: "
        f"{summary.get('reader_parse_success_rate', 0.0):.4f}"
    )
    print(
        "Avg invalid support IDs: "
        f"{summary.get('avg_num_invalid_support_ids', 0.0):.4f}"
    )

    print(f"Passage answer hit rate: {summary.get('passage_answer_hit_rate', 0.0):.4f}")
    print(f"Filtered triple answer hit rate: {summary.get('filtered_triple_answer_hit_rate', 0.0):.4f}")
    print(f"Evidence answer hit rate: {summary.get('evidence_answer_hit_rate', 0.0):.4f}")
    print(f"Gold ID overlap rate: {summary.get('gold_id_overlap_rate', 0.0):.4f}")
    print(
        "Question source attachment rate: "
        f"{summary.get('question_source_attachment_rate', 0.0):.4f}"
    )
    print(
        "Content-match evaluation rate: "
        f"{summary.get('content_match_evaluation_rate', 0.0):.4f}"
    )

    if "warning" in summary:
        print("WARNING:", summary["warning"])

    print(f"summary_path: {args.output_path}")

    if args.per_example_output_path:
        print(f"per_example_path: {args.per_example_output_path}")


if __name__ == "__main__":
    main()
