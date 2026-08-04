#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Remap/calibrate GRPO-v2 fusion-policy actions.

Purpose:
  The learned fusion policy may over-select a weaker action, for example keep3.
  This script reads the policy-selected action from an adaptive fusion output,
  optionally remaps it, then rebuilds evidence using base PPR + LLM-select files.

Example:
  keep3 -> keep2

This does NOT use gold labels.
It only uses:
  - base PPR retrieval file
  - LLM-selected retrieval file
  - existing adaptive fusion output containing selected actions
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Set


VALID_ACTIONS = {"pure_ppr", "pure_llm", "keep1", "keep2", "keep3", "keep4"}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_passages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        row.get("evidence_passages")
        or row.get("retrieved_passages")
        or row.get("passages")
        or []
    )


def set_passages(row: Dict[str, Any], passages: List[Dict[str, Any]]) -> None:
    if "evidence_passages" in row:
        row["evidence_passages"] = passages
    elif "retrieved_passages" in row:
        row["retrieved_passages"] = passages
    elif "passages" in row:
        row["passages"] = passages
    else:
        row["evidence_passages"] = passages


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


def alias_key(p: Dict[str, Any]) -> str:
    ids = sorted(passage_aliases(p))
    if ids:
        return "||".join(ids)
    return json.dumps(p, sort_keys=True, ensure_ascii=False)


def add_unique(out: List[Dict[str, Any]], p: Dict[str, Any], seen: Set[str]) -> None:
    key = alias_key(p)
    if key in seen:
        return
    seen.add(key)
    out.append(p)


def fuse_passages(
    action: str,
    base_passages: List[Dict[str, Any]],
    llm_passages: List[Dict[str, Any]],
    target_k: int,
) -> List[Dict[str, Any]]:

    if action == "pure_ppr":
        return list(base_passages)

    if action == "pure_llm":
        out = []
        seen = set()

        for p in llm_passages:
            add_unique(out, p, seen)

        for p in base_passages:
            add_unique(out, p, seen)

        return out

    if action.startswith("keep"):
        try:
            keep_n = int(action.replace("keep", ""))
        except Exception:
            keep_n = 2

        out = []
        seen = set()

        # 1. Keep top PPR fixed.
        for p in base_passages[:keep_n]:
            add_unique(out, p, seen)

        # 2. Fill remaining top-k slots from LLM-selected order.
        for p in llm_passages:
            if len(out) >= target_k:
                break
            add_unique(out, p, seen)

        # 3. Preserve remaining evidence pool from original PPR order.
        for p in base_passages:
            add_unique(out, p, seen)

        return out

    raise ValueError(f"Unknown action: {action}")


def parse_remap(s: str) -> Dict[str, str]:
    """
    Format:
      keep3:keep2
      keep3:keep2,pure_llm:keep1
    """
    mp = {}

    s = (s or "").strip()
    if not s:
        return mp

    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]

    for part in parts:
        if ":" not in part:
            raise ValueError(f"Bad remap item: {part}. Expected old:new")

        old, new = [x.strip() for x in part.split(":", 1)]

        if old not in VALID_ACTIONS:
            raise ValueError(f"Invalid old action: {old}")
        if new not in VALID_ACTIONS:
            raise ValueError(f"Invalid new action: {new}")

        mp[old] = new

    return mp


def get_policy_action(row: Dict[str, Any], prefer_raw_policy: bool = False) -> str:
    meta = row.get("metadata") or {}

    # From scripts/apply_fusion_policy.py
    fp = meta.get("fusion_policy") or {}
    if fp:
        a = fp.get("selected_action")
        if a:
            return str(a)

    # From scripts/apply_fusion_policy_confidence.py
    fpc = meta.get("fusion_policy_confidence") or {}
    if fpc:
        if prefer_raw_policy and fpc.get("raw_policy_action"):
            return str(fpc.get("raw_policy_action"))
        if fpc.get("chosen_action"):
            return str(fpc.get("chosen_action"))
        if fpc.get("raw_policy_action"):
            return str(fpc.get("raw_policy_action"))

    # From fallback/error metadata if present.
    if meta.get("fusion_policy_fallback_action"):
        return str(meta.get("fusion_policy_fallback_action"))

    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--llmselect_path", required=True)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--summary_path", default=None)

    ap.add_argument("--remap", default="keep3:keep2")
    ap.add_argument("--target_k", type=int, default=5)
    ap.add_argument("--default_action", default="keep2")
    ap.add_argument("--prefer_raw_policy", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        raise SystemExit(f"Output exists. Use --force: {args.output_path}")

    if args.default_action not in VALID_ACTIONS:
        raise ValueError(f"Invalid default_action: {args.default_action}")

    remap = parse_remap(args.remap)

    base_rows = load_jsonl(args.base_path)
    llm_rows = load_jsonl(args.llmselect_path)
    policy_rows = load_jsonl(args.policy_path)

    n = min(len(base_rows), len(llm_rows), len(policy_rows))
    if args.limit is not None:
        n = min(n, args.limit)

    out_rows = []

    original_counts = Counter()
    remapped_counts = Counter()
    actual_remap_counts = Counter()
    errors = 0
    missing_policy_action = 0

    for i in range(n):
        base_row = base_rows[i]
        llm_row = llm_rows[i]
        policy_row = policy_rows[i]

        new_row = dict(base_row)

        try:
            original_action = get_policy_action(
                policy_row,
                prefer_raw_policy=args.prefer_raw_policy,
            )

            if not original_action:
                missing_policy_action += 1
                original_action = args.default_action

            if original_action not in VALID_ACTIONS:
                original_action = args.default_action

            final_action = remap.get(original_action, original_action)

            base_passages = get_passages(base_row)
            llm_passages = get_passages(llm_row)

            fused = fuse_passages(
                action=final_action,
                base_passages=base_passages,
                llm_passages=llm_passages,
                target_k=args.target_k,
            )

            set_passages(new_row, fused)

            original_counts[original_action] += 1
            remapped_counts[final_action] += 1

            if original_action != final_action:
                actual_remap_counts[f"{original_action}->{final_action}"] += 1

            new_row.setdefault("metadata", {})
            new_row["metadata"]["fusion_policy_action_remap"] = {
                "policy_path": args.policy_path,
                "original_action": original_action,
                "final_action": final_action,
                "remap": remap,
                "target_k": args.target_k,
                "default_action": args.default_action,
                "prefer_raw_policy": args.prefer_raw_policy,
            }

        except Exception as e:
            errors += 1

            base_passages = get_passages(base_row)
            llm_passages = get_passages(llm_row)

            fused = fuse_passages(
                action=args.default_action,
                base_passages=base_passages,
                llm_passages=llm_passages,
                target_k=args.target_k,
            )

            set_passages(new_row, fused)

            original_counts["ERROR"] += 1
            remapped_counts[args.default_action] += 1

            new_row.setdefault("metadata", {})
            new_row["metadata"]["fusion_policy_action_remap_error"] = str(e)
            new_row["metadata"]["fusion_policy_action_remap_fallback"] = args.default_action

        out_rows.append(new_row)

        if (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{n}", flush=True)

    write_jsonl(out_rows, args.output_path)

    summary = {
        "base_path": args.base_path,
        "llmselect_path": args.llmselect_path,
        "policy_path": args.policy_path,
        "output_path": args.output_path,
        "num_rows": len(out_rows),
        "target_k": args.target_k,
        "remap": remap,
        "default_action": args.default_action,
        "prefer_raw_policy": args.prefer_raw_policy,
        "original_action_counts": dict(original_counts),
        "final_action_counts": dict(remapped_counts),
        "actual_remap_counts": dict(actual_remap_counts),
        "missing_policy_action": missing_policy_action,
        "num_errors": errors,
    }

    print(json.dumps(summary, indent=2))

    if args.summary_path:
        write_json(summary, args.summary_path)
        print("summary_path:", args.summary_path)

    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
