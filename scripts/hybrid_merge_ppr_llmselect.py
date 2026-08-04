import argparse
import json
import os
from typing import Any, Dict, List, Set


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_passages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return row.get("evidence_passages") or row.get("retrieved_passages") or row.get("passages") or []


def set_passages(row: Dict[str, Any], passages: List[Dict[str, Any]]) -> None:
    if "evidence_passages" in row:
        row["evidence_passages"] = passages
    elif "retrieved_passages" in row:
        row["retrieved_passages"] = passages
    elif "passages" in row:
        row["passages"] = passages
    else:
        row["evidence_passages"] = passages


def aliases(p: Dict[str, Any]) -> Set[str]:
    ids = set()

    for k in ["id", "passage_id", "node_id", "idx", "fallback_idx"]:
        if p.get(k) is not None:
            ids.add(str(p.get(k)))

    meta = p.get("metadata") or {}
    for k in ["fallback_idx", "idx", "passage_idx", "source_idx", "corpus_idx", "passage_id", "node_id"]:
        if meta.get(k) is not None:
            ids.add(str(meta.get(k)))

    return ids


def add_unique(out, p, seen):
    a = aliases(p)
    key = tuple(sorted(a)) if a else json.dumps(p, sort_keys=True)
    if key in seen:
        return
    seen.add(key)
    out.append(p)


def merge_one(base_row, llm_row, keep_ppr_top_n: int, target_top_k: int):
    base_passages = get_passages(base_row)
    llm_passages = get_passages(llm_row)

    out = []
    seen = set()

    # 1. Keep top PPR evidence fixed.
    for p in base_passages[:keep_ppr_top_n]:
        add_unique(out, p, seen)

    # 2. Fill with LLM-selected/reranked evidence.
    for p in llm_passages:
        if len(out) >= target_top_k:
            break
        add_unique(out, p, seen)

    # 3. Fill remaining slots with original PPR order.
    for p in base_passages:
        if len(out) >= len(base_passages):
            break
        add_unique(out, p, seen)

    new_row = dict(base_row)
    set_passages(new_row, out)

    new_row.setdefault("metadata", {})
    new_row["metadata"]["hybrid_ppr_llmselect"] = {
        "keep_ppr_top_n": keep_ppr_top_n,
        "target_top_k": target_top_k,
    }

    return new_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--llmselect_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--keep_ppr_top_n", type=int, default=2)
    ap.add_argument("--target_top_k", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        raise SystemExit(f"Output exists. Use --force: {args.output_path}")

    base_rows = load_jsonl(args.base_path)
    llm_rows = load_jsonl(args.llmselect_path)

    n = min(len(base_rows), len(llm_rows))
    out_rows = []

    for i in range(n):
        out_rows.append(
            merge_one(
                base_rows[i],
                llm_rows[i],
                keep_ppr_top_n=args.keep_ppr_top_n,
                target_top_k=args.target_top_k,
            )
        )

    write_jsonl(out_rows, args.output_path)
    print("saved:", args.output_path)
    print("num rows:", len(out_rows))


if __name__ == "__main__":
    main()
