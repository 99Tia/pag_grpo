#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Conservative chain-aware evidence reranker.

Input:
  retrieval JSONL from run_agent_retrieval.py

Output:
  retrieval JSONL with evidence_passages reranked.

This script does NOT use gold answers or gold passage IDs for scoring.
It only uses retrieved passages, filtered triples, question text, and original ranks/scores.
"""

import argparse
import copy
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "from", "by", "with",
    "and", "or", "as", "at", "is", "was", "were", "are", "be", "been",
    "when", "where", "who", "whom", "what", "which", "whose", "how",
    "did", "does", "do", "had", "has", "have", "that", "this", "it",
    "their", "his", "her", "its", "into", "about", "after", "before",
    "between", "among", "during", "also", "not", "but", "than", "then",
}


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


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").lower()).strip()


def tokenize(x: Any) -> List[str]:
    text = norm_text(x)
    toks = re.findall(r"[a-z0-9][a-z0-9\-']+", text)
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def token_set(x: Any) -> Set[str]:
    return set(tokenize(x))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def safe_get_text(p: Dict[str, Any]) -> str:
    return (
        p.get("text")
        or p.get("passage_text")
        or p.get("content")
        or p.get("paragraph_text")
        or ""
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


def get_evidence_passages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        row.get("evidence_passages")
        or row.get("retrieved_passages")
        or row.get("passages")
        or []
    )


def set_evidence_passages(row: Dict[str, Any], passages: List[Dict[str, Any]]) -> None:
    if "evidence_passages" in row:
        row["evidence_passages"] = passages
    elif "retrieved_passages" in row:
        row["retrieved_passages"] = passages
    elif "passages" in row:
        row["passages"] = passages
    else:
        row["evidence_passages"] = passages


def get_filtered_triples(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    triples = list(row.get("filtered_triples") or [])

    # Fallback: recover triples from step observations if top-level missing.
    if not triples:
        for step in row.get("steps") or []:
            obs = step.get("observation") or {}
            sr = obs.get("search_result") or {}
            triples.extend(sr.get("filtered_triples") or [])

    # Deduplicate roughly by text/source.
    seen = set()
    out = []
    for t in triples:
        key = json.dumps(t, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def triple_text(t: Dict[str, Any]) -> str:
    parts = []
    for k in [
        "subject",
        "predicate",
        "object",
        "relation",
        "head",
        "tail",
        "text",
        "triple_text",
        "normalized_triple",
    ]:
        v = t.get(k)
        if v:
            parts.append(str(v))
    return " ".join(parts)


def triple_source_aliases(t: Dict[str, Any]) -> Set[str]:
    ids = set()

    for k in [
        "source_passage_id",
        "passage_id",
        "source_id",
        "source_idx",
        "fallback_idx",
        "idx",
    ]:
        if t.get(k) is not None:
            ids.add(str(t.get(k)))

    meta = t.get("metadata") or {}
    for k in [
        "source_passage_id",
        "passage_id",
        "source_id",
        "source_idx",
        "fallback_idx",
        "idx",
    ]:
        if meta.get(k) is not None:
            ids.add(str(meta.get(k)))

    # Some filtered triples wrap an original/candidate triple.
    for wrapper_key in ["triple", "candidate", "raw_triple", "original_triple"]:
        inner = t.get(wrapper_key)
        if isinstance(inner, dict):
            ids |= triple_source_aliases(inner)

    return ids


def original_score(p: Dict[str, Any]) -> float:
    for k in ["score", "ppr_score", "retrieval_score", "final_score"]:
        if p.get(k) is not None:
            try:
                return float(p[k])
            except Exception:
                pass

    meta = p.get("metadata") or {}
    for k in ["score", "ppr_score", "retrieval_score", "final_score"]:
        if meta.get(k) is not None:
            try:
                return float(meta[k])
            except Exception:
                pass

    return 0.0


def normalize_values(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def rerank_one(
    row: Dict[str, Any],
    top_pool: int = 10,
    keep_top_n: int = 1,
    mmr_lambda: float = 0.78,
    source_boost_weight: float = 0.22,
    entity_weight: float = 0.16,
    question_weight: float = 0.10,
    score_weight: float = 0.18,
    rank_weight: float = 0.34,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    passages = get_evidence_passages(row)
    if len(passages) <= 1:
        return passages, {"rerank_applied": False, "reason": "too_few_passages"}

    pool = copy.deepcopy(passages[:top_pool])
    tail = copy.deepcopy(passages[top_pool:])

    question = row.get("question") or row.get("query") or ""
    q_tokens = token_set(question)

    filtered_triples = get_filtered_triples(row)
    triple_tokens = token_set(" ".join(triple_text(t) for t in filtered_triples))

    source_counts = Counter()
    for t in filtered_triples:
        for sid in triple_source_aliases(t):
            source_counts[sid] += 1

    raw_scores = [original_score(p) for p in pool]
    norm_scores = normalize_values(raw_scores)

    features = []
    for i, p in enumerate(pool):
        aliases = passage_aliases(p)
        p_text = safe_get_text(p)
        p_tokens = token_set(p_text)

        rank_prior = 1.0 / math.sqrt(i + 1)

        source_hit_count = sum(source_counts.get(a, 0) for a in aliases)
        source_boost = min(1.0, source_hit_count / 2.0)

        entity_overlap = jaccard(p_tokens, triple_tokens)
        q_overlap = jaccard(p_tokens, q_tokens)

        base = (
            rank_weight * rank_prior
            + score_weight * norm_scores[i]
            + source_boost_weight * source_boost
            + entity_weight * entity_overlap
            + question_weight * q_overlap
        )

        features.append({
            "rank": i + 1,
            "aliases": sorted(aliases),
            "tokens": p_tokens,
            "rank_prior": rank_prior,
            "norm_score": norm_scores[i],
            "source_boost": source_boost,
            "source_hit_count": source_hit_count,
            "entity_overlap": entity_overlap,
            "question_overlap": q_overlap,
            "base_score": base,
        })

    selected = []
    selected_idx = set()

    # Conservative anchor: keep the very top evidence fixed.
    for i in range(min(keep_top_n, len(pool))):
        selected.append(i)
        selected_idx.add(i)

    # MMR selection for remaining positions.
    while len(selected) < len(pool):
        best_i = None
        best_score = -1e9

        for i in range(len(pool)):
            if i in selected_idx:
                continue

            redundancy = 0.0
            for j in selected:
                redundancy = max(redundancy, jaccard(features[i]["tokens"], features[j]["tokens"]))

            mmr_score = mmr_lambda * features[i]["base_score"] - (1.0 - mmr_lambda) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i

        selected.append(best_i)
        selected_idx.add(best_i)

    reranked_pool = [pool[i] for i in selected]

    # Attach debug metadata without destroying original fields.
    for new_rank, old_i in enumerate(selected, start=1):
        meta = reranked_pool[new_rank - 1].setdefault("metadata", {})
        meta["chain_rerank_old_rank"] = features[old_i]["rank"]
        meta["chain_rerank_new_rank"] = new_rank
        meta["chain_rerank_base_score"] = features[old_i]["base_score"]
        meta["chain_rerank_source_boost"] = features[old_i]["source_boost"]
        meta["chain_rerank_source_hit_count"] = features[old_i]["source_hit_count"]
        meta["chain_rerank_entity_overlap"] = features[old_i]["entity_overlap"]
        meta["chain_rerank_question_overlap"] = features[old_i]["question_overlap"]

    debug = {
        "rerank_applied": True,
        "top_pool": top_pool,
        "keep_top_n": keep_top_n,
        "mmr_lambda": mmr_lambda,
        "old_order": [features[i]["rank"] for i in range(len(pool))],
        "new_order_old_ranks": [features[i]["rank"] for i in selected],
    }

    return reranked_pool + tail, debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--top_pool", type=int, default=10)
    ap.add_argument("--keep_top_n", type=int, default=1)
    ap.add_argument("--mmr_lambda", type=float, default=0.78)
    ap.add_argument("--source_boost_weight", type=float, default=0.22)
    ap.add_argument("--entity_weight", type=float, default=0.16)
    ap.add_argument("--question_weight", type=float, default=0.10)
    ap.add_argument("--score_weight", type=float, default=0.18)
    ap.add_argument("--rank_weight", type=float, default=0.34)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        raise SystemExit(f"Output exists. Use --force: {args.output_path}")

    rows = load_jsonl(args.input_path)
    out_rows = []

    for row in rows:
        new_row = copy.deepcopy(row)
        reranked, debug = rerank_one(
            new_row,
            top_pool=args.top_pool,
            keep_top_n=args.keep_top_n,
            mmr_lambda=args.mmr_lambda,
            source_boost_weight=args.source_boost_weight,
            entity_weight=args.entity_weight,
            question_weight=args.question_weight,
            score_weight=args.score_weight,
            rank_weight=args.rank_weight,
        )
        set_evidence_passages(new_row, reranked)
        new_row.setdefault("metadata", {})
        new_row["metadata"]["chain_reranker"] = debug
        out_rows.append(new_row)

    write_jsonl(out_rows, args.output_path)

    print("saved:", args.output_path)
    print("num rows:", len(out_rows))


if __name__ == "__main__":
    main()
