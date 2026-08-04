#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM evidence selector.

Input:
  retrieval JSONL with top-k evidence passages.

Output:
  retrieval JSONL with evidence_passages reordered so the LLM-selected
  passages are first.

This does NOT use gold answers or gold passage ids.
It only uses question, retrieved passages, and filtered triples.
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


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


def passage_text(p: Dict[str, Any]) -> str:
    return (
        p.get("text")
        or p.get("passage_text")
        or p.get("paragraph_text")
        or p.get("content")
        or ""
    )


def get_filtered_triples(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    triples = list(row.get("filtered_triples") or [])

    if not triples:
        for step in row.get("steps") or []:
            obs = step.get("observation") or {}
            sr = obs.get("search_result") or {}
            triples.extend(sr.get("filtered_triples") or [])

    seen = set()
    out = []
    for t in triples:
        key = json.dumps(t, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def triple_to_text(t: Dict[str, Any]) -> str:
    parts = []
    for k in ["subject", "predicate", "object", "relation", "head", "tail", "text", "triple_text"]:
        if t.get(k):
            parts.append(str(t[k]))
    return " | ".join(parts)


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found.")
    return json.loads(m.group(0))


def build_prompt(row: Dict[str, Any], passages: List[Dict[str, Any]], top_pool: int, select_k: int) -> str:
    q = row.get("question") or row.get("query") or ""

    triples = get_filtered_triples(row)[:30]
    triple_block = ""
    if triples:
        triple_lines = []
        for i, t in enumerate(triples, start=1):
            tx = triple_to_text(t)
            if tx:
                triple_lines.append(f"{i}. {tx[:300]}")
        triple_block = "\n".join(triple_lines)

    passage_lines = []
    for i, p in enumerate(passages[:top_pool], start=1):
        txt = passage_text(p).replace("\n", " ")
        title = p.get("title") or (p.get("metadata") or {}).get("title") or ""
        score = p.get("score") or (p.get("metadata") or {}).get("score") or ""
        passage_lines.append(
            f"[{i}] title: {title}\nscore: {score}\ntext: {txt[:1200]}"
        )

    return f"""You are an evidence selector for multi-hop question answering.

Your task:
Select exactly {select_k} passages from the candidate list that best support answering the question.
Prefer a set of passages that together covers the full reasoning chain:
- bridge entity evidence
- final answer evidence
- passages connected to filtered triples
- non-redundant evidence

Do not answer the question.
Do not use outside knowledge.
Only select passage numbers from the provided candidate passages.

Question:
{q}

Filtered triples:
{triple_block if triple_block else "None"}

Candidate passages:
{chr(10).join(passage_lines)}

Return only valid JSON:
{{
  "selected_indices": [1, 2, 3, 4, 5],
  "reason": "short reason"
}}
"""


def call_llm(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON. You select evidence passages for multi-hop QA.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            return extract_json(text)
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"LLM failed after retries: {last_err}")


def reorder_by_selection(
    passages: List[Dict[str, Any]],
    selected_indices: List[int],
    top_pool: int,
) -> List[Dict[str, Any]]:
    pool = passages[:top_pool]
    tail = passages[top_pool:]

    selected_zero = []
    seen = set()

    for idx in selected_indices:
        try:
            z = int(idx) - 1
        except Exception:
            continue
        if 0 <= z < len(pool) and z not in seen:
            selected_zero.append(z)
            seen.add(z)

    # Fill missing selected slots by original order.
    for z in range(len(pool)):
        if z not in seen:
            selected_zero.append(z)
            seen.add(z)

    return [pool[z] for z in selected_zero] + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--top_pool", type=int, default=10)
    ap.add_argument("--select_k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        raise SystemExit(f"Output exists. Use --force: {args.output_path}")

    api_key = os.environ.get(args.api_key_env, "dummy")
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    rows = load_jsonl(args.input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    out = []

    for n, row in enumerate(rows, start=1):
        passages = get_passages(row)
        new_row = dict(row)

        if len(passages) <= args.select_k:
            out.append(new_row)
            continue

        prompt = build_prompt(row, passages, args.top_pool, args.select_k)

        try:
            obj = call_llm(
                client=client,
                model_name=args.model_name,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            selected = obj.get("selected_indices") or []
            reason = obj.get("reason") or ""
            reordered = reorder_by_selection(passages, selected, args.top_pool)

            # add rank metadata
            for new_rank, p in enumerate(reordered[:args.top_pool], start=1):
                p.setdefault("metadata", {})
                p["metadata"]["llm_selector_new_rank"] = new_rank

            set_passages(new_row, reordered)
            new_row.setdefault("metadata", {})
            new_row["metadata"]["llm_evidence_selector"] = {
                "selected_indices": selected,
                "reason": reason,
                "top_pool": args.top_pool,
                "select_k": args.select_k,
            }

        except Exception as e:
            new_row.setdefault("metadata", {})
            new_row["metadata"]["llm_evidence_selector_error"] = str(e)

        out.append(new_row)

        if n % 10 == 0:
            print(f"processed {n}/{len(rows)}", flush=True)

    write_jsonl(out, args.output_path)
    print("saved:", args.output_path)
    print("num rows:", len(out))


if __name__ == "__main__":
    main()
