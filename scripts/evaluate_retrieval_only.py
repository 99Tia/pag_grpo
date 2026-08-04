import argparse
import json
import os
from typing import Any, Dict, List, Set


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_passages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        row.get("evidence_passages")
        or row.get("retrieved_passages")
        or row.get("passages")
        or []
    )


def passage_aliases(p: Dict[str, Any]) -> Set[str]:
    ids = set()

    for k in ["id", "passage_id", "node_id", "idx", "fallback_idx"]:
        if p.get(k) is not None:
            ids.add(str(p.get(k)))

    meta = p.get("metadata") or {}
    for k in [
        "fallback_idx",
        "idx",
        "passage_idx",
        "source_idx",
        "corpus_idx",
        "passage_id",
        "node_id",
    ]:
        if meta.get(k) is not None:
            ids.add(str(meta.get(k)))

    return ids


def get_gold_ids(row: Dict[str, Any]) -> Set[str]:
    gold = (
        row.get("gold_passage_ids")
        or row.get("supporting_passage_ids")
        or row.get("support_ids")
        or []
    )
    return {str(x) for x in gold if x is not None}


def eval_row(row: Dict[str, Any], k: int) -> Dict[str, float]:
    gold = get_gold_ids(row)
    passages = get_passages(row)

    if not gold:
        return {
            "has_gold": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "full_support": 0.0,
            "num_gold": 0.0,
            "num_retrieved": float(min(k, len(passages))),
            "num_hits": 0.0,
        }

    hits = set()
    retrieved_count = min(k, len(passages))

    for p in passages[:k]:
        ids = passage_aliases(p)
        hits |= (gold & ids)

    return {
        "has_gold": 1.0,
        "recall": len(hits) / max(1, len(gold)),
        "precision": len(hits) / max(1, retrieved_count),
        "full_support": 1.0 if gold <= hits else 0.0,
        "num_gold": float(len(gold)),
        "num_retrieved": float(retrieved_count),
        "num_hits": float(len(hits)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_path", required=True)
    ap.add_argument("--output_path", default=None)
    ap.add_argument("--per_example_output_path", default=None)
    ap.add_argument("--k_list", default="1,5,10")
    args = ap.parse_args()

    rows = load_jsonl(args.retrieval_path)
    k_list = [int(x) for x in args.k_list.replace(",", " ").split()]

    summary = {
        "retrieval_path": args.retrieval_path,
        "num_examples": len(rows),
    }

    rows_with_gold = [r for r in rows if get_gold_ids(r)]
    summary["num_examples_with_gold_support"] = len(rows_with_gold)

    avg_evidence = sum(len(get_passages(r)) for r in rows) / max(1, len(rows))
    summary["avg_evidence_passages"] = avg_evidence

    per_rows = []

    for row_idx, row in enumerate(rows):
        qid = row.get("id") or row.get("question_id") or row.get("qid")
        per = {
            "row_idx": row_idx,
            "id": qid,
            "question": row.get("question") or row.get("query"),
            "gold_passage_ids": sorted(get_gold_ids(row)),
            "num_evidence_passages": len(get_passages(row)),
        }

        for k in k_list:
            m = eval_row(row, k)
            per[f"recall@{k}"] = m["recall"]
            per[f"precision@{k}"] = m["precision"]
            per[f"full_support@{k}"] = m["full_support"]
            per[f"num_hits@{k}"] = m["num_hits"]

        per_rows.append(per)

    denom = max(1, len(rows_with_gold))

    for k in k_list:
        recall_sum = 0.0
        precision_sum = 0.0
        full_sum = 0.0

        for row in rows_with_gold:
            m = eval_row(row, k)
            recall_sum += m["recall"]
            precision_sum += m["precision"]
            full_sum += m["full_support"]

        summary[f"recall@{k}"] = recall_sum / denom
        summary[f"precision@{k}"] = precision_sum / denom
        summary[f"full_support@{k}"] = full_sum / denom

    print(json.dumps(summary, indent=2))

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("summary_path:", args.output_path)

    if args.per_example_output_path:
        os.makedirs(os.path.dirname(args.per_example_output_path), exist_ok=True)
        with open(args.per_example_output_path, "w", encoding="utf-8") as f:
            for r in per_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("per_example_path:", args.per_example_output_path)


if __name__ == "__main__":
    main()
