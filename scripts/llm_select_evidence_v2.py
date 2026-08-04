import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Set

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


def passage_title(p: Dict[str, Any]) -> str:
    meta = p.get("metadata") or {}
    return str(p.get("title") or meta.get("title") or "")


def passage_score(p: Dict[str, Any]) -> str:
    meta = p.get("metadata") or {}
    score = p.get("score")
    if score is None:
        score = meta.get("score")
    if score is None:
        score = meta.get("ppr_score")
    if score is None:
        return ""
    return str(score)


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
    if t.get("subject") and (t.get("predicate") or t.get("relation")) and t.get("object"):
        pred = t.get("predicate") or t.get("relation")
        return f"{t.get('subject')} | {pred} | {t.get('object')}"

    if t.get("head") and (t.get("predicate") or t.get("relation")) and t.get("tail"):
        pred = t.get("predicate") or t.get("relation")
        return f"{t.get('head')} | {pred} | {t.get('tail')}"

    parts = []
    for k in ["subject", "predicate", "object", "relation", "head", "tail", "text", "triple_text"]:
        if t.get(k):
            parts.append(str(t[k]))

    return " | ".join(parts)



def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I)
    text = re.sub(r"```$", "", text.strip())

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in LLM output.")

    return json.loads(m.group(0))


def normalize_indices(indices: Any, top_pool: int, select_k: int) -> List[int]:
    if not isinstance(indices, list):
        return []

    out = []
    seen = set()

    for x in indices:
        try:
            idx = int(x)
        except Exception:
            continue

        if 1 <= idx <= top_pool and idx not in seen:
            seen.add(idx)
            out.append(idx)

        if len(out) >= select_k:
            break

    return out


def build_prompt(
    row: Dict[str, Any],
    passages: List[Dict[str, Any]],
    top_pool: int,
    select_k: int,
    max_passage_chars: int,
    max_triples: int,
) -> str:
    question = row.get("question") or row.get("query") or ""

    triples = get_filtered_triples(row)[:max_triples]
    triple_lines = []

    for i, t in enumerate(triples, start=1):
        tx = triple_to_text(t)
        if tx:
            triple_lines.append(f"{i}. {tx[:300]}")

    passage_lines = []

    for i, p in enumerate(passages[:top_pool], start=1):
        txt = passage_text(p).replace("\n", " ")
        txt = re.sub(r"\s+", " ", txt).strip()

        title = passage_title(p)
        score = passage_score(p)
        aliases = sorted(passage_aliases(p))

        alias_text = ", ".join(aliases[:4]) if aliases else ""

        passage_lines.append(
            f"[{i}]\n"
            f"title: {title}\n"
            f"score: {score}\n"
            f"ids: {alias_text}\n"
            f"text: {txt[:max_passage_chars]}"
        )

    return f"""You are a chain-aware evidence selector for multi-hop question answering.

You must select exactly {select_k} passages from the candidate list.

Important:
- Do NOT answer the question.
- Do NOT use outside knowledge.
- Use only the provided candidate passages and filtered triples.
- Your goal is not to pick five individually similar passages.
- Your goal is to pick a small evidence set that together supports the full reasoning chain.

Selection rules:
1. First infer the likely reasoning hops required by the question.
2. Prefer passages that connect entities across hops, not just passages that mention the final answer.
3. Prefer a complete chain: bridge evidence + intermediate entity evidence + final answer evidence.
4. Avoid redundant passages unless redundancy is needed to connect the chain.
5. If a passage is highly ranked by PPR but does not support any needed hop, do not select it only because of the score.
6. If a lower-ranked passage completes a missing hop, promote it.
7. Return selected passage numbers in the order they should be read for reasoning.

Question:
{question}

Filtered triples:
{chr(10).join(triple_lines) if triple_lines else "None"}

Candidate passages:
{chr(10).join(passage_lines)}

Return only valid JSON with this schema:
{{
  "inferred_hops": [
    "hop 1 description",
    "hop 2 description"
  ],
  "passage_roles": [
    {{"index": 1, "role": "bridge/final/redundant/irrelevant", "supports_hop": "short explanation"}}
  ],
  "selected_indices": [1, 2, 3, 4, 5],
  "chain_reason": "short explanation of why these passages cover the full chain",
  "missing_evidence_warning": "None, or short warning if the provided candidates do not fully cover the chain"
}}
"""


def call_llm(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
    sleep_base: float = 1.0,
) -> Dict[str, Any]:
    last_err = None

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only valid JSON. "
                            "You are selecting chain-complete evidence passages for multi-hop QA."
                        ),
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
            time.sleep(sleep_base + attempt)

    raise RuntimeError(f"LLM failed after {retries} retries: {last_err}")


def reorder_by_selection(
    passages: List[Dict[str, Any]],
    selected_indices: List[int],
    top_pool: int,
) -> List[Dict[str, Any]]:
    pool = passages[:top_pool]
    tail = passages[top_pool:]

    out = []
    seen = set()

    for idx in selected_indices:
        z = idx - 1
        if 0 <= z < len(pool):
            key = alias_key(pool[z])
            if key not in seen:
                seen.add(key)
                out.append(pool[z])

    for p in pool:
        key = alias_key(p)
        if key not in seen:
            seen.add(key)
            out.append(p)

    for p in tail:
        key = alias_key(p)
        if key not in seen:
            seen.add(key)
            out.append(p)

    return out


def add_rank_metadata(
    passages: List[Dict[str, Any]],
    selected_indices: List[int],
    top_pool: int,
) -> None:
    selected_set = set(selected_indices)

    for new_rank, p in enumerate(passages[:top_pool], start=1):
        p.setdefault("metadata", {})
        p["metadata"]["llm_selector_v2_new_rank"] = new_rank
        p["metadata"]["llm_selector_v2_selected"] = new_rank <= len(selected_set)



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)

    ap.add_argument("--base_url", required=True)
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY")
    ap.add_argument("--model_name", required=True)

    ap.add_argument("--top_pool", type=int, default=15)
    ap.add_argument("--select_k", type=int, default=5)
    ap.add_argument("--max_passage_chars", type=int, default=900)
    ap.add_argument("--max_triples", type=int, default=30)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--retries", type=int, default=3)

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
    error_count = 0
    short_passage_count = 0

    for n, row in enumerate(rows, start=1):
        passages = get_passages(row)
        new_row = dict(row)

        if len(passages) <= args.select_k:
            short_passage_count += 1
            new_row.setdefault("metadata", {})
            new_row["metadata"]["llm_evidence_selector_v2"] = {
                "skipped": True,
                "reason": "num_passages <= select_k",
                "top_pool": args.top_pool,
                "select_k": args.select_k,
            }
            out.append(new_row)
            continue

        effective_top_pool = min(args.top_pool, len(passages))

        prompt = build_prompt(
            row=row,
            passages=passages,
            top_pool=effective_top_pool,
            select_k=args.select_k,
            max_passage_chars=args.max_passage_chars,
            max_triples=args.max_triples,
        )

        try:
            obj = call_llm(
                client=client,
                model_name=args.model_name,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
            )

            selected = normalize_indices(
                obj.get("selected_indices") or [],
                top_pool=effective_top_pool,
                select_k=args.select_k,
            )

            # If LLM returned too few indices, fill by original rank.
            if len(selected) < args.select_k:
                seen = set(selected)
                for idx in range(1, effective_top_pool + 1):
                    if idx not in seen:
                        selected.append(idx)
                        seen.add(idx)
                    if len(selected) >= args.select_k:
                        break

            reordered = reorder_by_selection(
                passages=passages,
                selected_indices=selected,
                top_pool=effective_top_pool,
            )

            add_rank_metadata(
                passages=reordered,
                selected_indices=selected,
                top_pool=effective_top_pool,
            )

            set_passages(new_row, reordered)

            new_row.setdefault("metadata", {})
            new_row["metadata"]["llm_evidence_selector_v2"] = {
                "selected_indices": selected,
                "inferred_hops": obj.get("inferred_hops") or [],
                "passage_roles": obj.get("passage_roles") or [],
                "chain_reason": obj.get("chain_reason") or "",
                "missing_evidence_warning": obj.get("missing_evidence_warning") or "",
                "top_pool": effective_top_pool,
                "select_k": args.select_k,
                "max_passage_chars": args.max_passage_chars,
                "max_triples": args.max_triples,
            }

        except Exception as e:
            error_count += 1

            new_row.setdefault("metadata", {})
            new_row["metadata"]["llm_evidence_selector_v2_error"] = str(e)
            new_row["metadata"]["llm_evidence_selector_v2"] = {
                "selected_indices": [],
                "top_pool": effective_top_pool,
                "select_k": args.select_k,
                "fallback": "original_order",
            }

        out.append(new_row)

        if n % 10 == 0:
            print(f"processed {n}/{len(rows)} errors={error_count}", flush=True)

    write_jsonl(out, args.output_path)

    summary = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "num_rows": len(out),
        "top_pool": args.top_pool,
        "select_k": args.select_k,
        "max_passage_chars": args.max_passage_chars,
        "max_triples": args.max_triples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "error_count": error_count,
        "short_passage_count": short_passage_count,
    }

    summary_path = args.output_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print("saved:", args.output_path)
    print("summary_path:", summary_path)


if __name__ == "__main__":
    main()