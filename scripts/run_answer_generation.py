from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from ppr_agent.openie_extractor import OpenIEExtractorConfig, build_backend  # noqa: E402


logger = logging.getLogger(__name__)


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
        for key in ["data", "examples", "trajectories", "results"]:
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unsupported input format: {path}")


def save_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    try:
        return load_json_or_jsonl(path)
    except Exception:
        return []


def existing_question_ids(rows: Sequence[Dict[str, Any]]) -> set:
    ids = set()

    for row in rows:
        qid = row.get("question_id") or row.get("id")
        if qid is not None:
            ids.add(str(qid))

    return ids


def safe_get(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def unwrap_trajectory_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}

    trajectory = row.get("trajectory")

    if isinstance(trajectory, dict):
        merged = dict(trajectory)

        for key in [
            "question_id",
            "question",
            "gold_answers",
            "gold_passage_ids",
            "controller_final_answer",
            "answer_for_reward",
            "answer_source",
        ]:
            if key in row and key not in merged:
                merged[key] = row[key]

        return merged

    return row

def as_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [str(x) for x in value]

    return [str(value)]


def truncate_text(text: Any, max_chars: int) -> str:
    text = str(text or "").strip()

    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + " ..."

    return text


def normalize_generation_output(raw: Any) -> str:
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw

    if isinstance(raw, list):
        if not raw:
            return ""

        first = raw[0]

        if isinstance(first, str):
            return first

        if isinstance(first, dict):
            for key in ["text", "output", "response", "content"]:
                if key in first:
                    return str(first[key])
            return json.dumps(first, ensure_ascii=False)

        if hasattr(first, "outputs"):
            try:
                return str(first.outputs[0].text)
            except Exception:
                return str(first)

        return str(first)

    if isinstance(raw, dict):
        for key in ["text", "output", "response", "content"]:
            if key in raw:
                return str(raw[key])
        return json.dumps(raw, ensure_ascii=False)

    if hasattr(raw, "outputs"):
        try:
            return str(raw.outputs[0].text)
        except Exception:
            return str(raw)

    return str(raw)

def normalize_passage(raw: Any, fallback_rank: int) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None

        return {
            "passage_id": f"unknown-{fallback_rank}",
            "title": None,
            "text": text,
            "score": None,
            "rank": fallback_rank,
            "metadata": {},
        }

    if not isinstance(raw, dict):
        return None

    text = safe_get(raw, ["text", "passage", "content", "body", "paragraph"], "")
    text = str(text).strip()

    if not text:
        return None

    passage_id = safe_get(
        raw,
        ["passage_id", "id", "idx", "chunk_id", "node_id"],
        f"unknown-{fallback_rank}",
    )

    return {
        "passage_id": str(passage_id),
        "title": safe_get(raw, ["title", "name"], None),
        "text": text,
        "score": safe_get(raw, ["score", "ppr_score", "retrieval_score"], None),
        "rank": safe_get(raw, ["rank"], fallback_rank),
        "metadata": raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
    }


def collect_passages_from_steps(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    steps = row.get("steps", [])
    if not isinstance(steps, list):
        return output

    for step in steps:
        if not isinstance(step, dict):
            continue

        obs = step.get("observation") or {}
        if not isinstance(obs, dict):
            continue

        search_result = obs.get("search_result") or {}
        if not isinstance(search_result, dict):
            continue

        passages = search_result.get("passages") or []
        if not isinstance(passages, list):
            continue

        for item in passages:
            p = normalize_passage(item, fallback_rank=len(output) + 1)
            if p is not None:
                output.append(p)

    return output


def collect_evidence_passages(
    row: Dict[str, Any],
    top_k: int,
    max_passage_chars: int,
) -> List[Dict[str, Any]]:
    candidates = row.get("evidence_passages")

    if not isinstance(candidates, list) or not candidates:
        candidates = collect_passages_from_steps(row)

    output: List[Dict[str, Any]] = []
    seen = set()

    for i, item in enumerate(candidates):
        p = normalize_passage(item, fallback_rank=i + 1)
        if p is None:
            continue

        key = p["passage_id"]

        if key in seen:
            continue

        seen.add(key)

        p["text"] = truncate_text(p["text"], max_passage_chars)
        output.append(p)

        if len(output) >= top_k:
            break

    return output


def collect_filtered_triples(
    row: Dict[str, Any],
    top_k: int,
) -> List[Any]:
    triples = row.get("filtered_triples")

    if isinstance(triples, list) and triples:
        return triples[:top_k]

    output: List[Any] = []
    seen = set()

    steps = row.get("steps", [])
    if not isinstance(steps, list):
        return output

    for step in steps:
        obs = step.get("observation") if isinstance(step, dict) else None
        if not isinstance(obs, dict):
            continue

        sr = obs.get("search_result")
        if not isinstance(sr, dict):
            continue

        vals = sr.get("filtered_triples") or []
        if not isinstance(vals, list):
            continue

        for t in vals:
            key = json.dumps(t, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue

            seen.add(key)
            output.append(t)

            if len(output) >= top_k:
                return output

    return output


def format_triple_for_prompt(triple_obj: Any, idx: int) -> str:
    score = None
    triple = triple_obj

    if isinstance(triple_obj, dict):
        score = triple_obj.get("original_score") or triple_obj.get("score")
        triple = triple_obj.get("triple", triple_obj)

    if isinstance(triple, dict):
        subj = triple.get("subject") or triple.get("subj") or triple.get("head") or ""
        pred = triple.get("predicate") or triple.get("relation") or triple.get("pred") or ""
        obj = triple.get("object") or triple.get("obj") or triple.get("tail") or ""

        text = f"({subj}, {pred}, {obj})"
    elif isinstance(triple, list) and len(triple) >= 3:
        text = f"({triple[0]}, {triple[1]}, {triple[2]})"
    else:
        text = str(triple)

    if score is not None:
        return f"[T{idx} score={score}] {text}"

    return f"[T{idx}] {text}"


def build_answer_messages(
    question: str,
    evidence_passages: Sequence[Dict[str, Any]],
    filtered_triples: Sequence[Any],
) -> List[Dict[str, str]]:
    triple_lines: List[str] = []
    for i, triple in enumerate(filtered_triples, start=1):
        triple_lines.append(format_triple_for_prompt(triple, i))

    if triple_lines:
        triples_text = "\n".join(triple_lines)
    else:
        triples_text = "No filtered triples were saved."

    passage_blocks: List[str] = []
    for i, p in enumerate(evidence_passages, start=1):
        pid = p.get("passage_id", f"passage-{i}")
        title = p.get("title")
        text = p.get("text", "")

        header = f"[P{i}] passage_id={pid}"
        if title:
            header += f" | title={title}"

        passage_blocks.append(f"{header}\n{text}")

    if passage_blocks:
        passages_text = "\n\n".join(passage_blocks)
    else:
        passages_text = "No evidence passages were retrieved."

    system_prompt = """You are a precise multi-hop question-answering model.

You must answer using ONLY the provided filtered triples and evidence passages.
Do not invent facts.
If the evidence does not contain the answer, answer exactly: I don't know.
Return only valid JSON.
"""

    user_prompt = f"""Question:
{question}

Filtered triples:
{triples_text}

Evidence passages:
{passages_text}

Return ONLY valid JSON with this schema:
{{
  "answer": "short final answer string",
  "supporting_passage_ids": ["passage_id"],
  "supporting_triples": ["triple ids or short triple text"],
  "confidence": 0.0
}}

Rules:
- The answer should be the shortest correct answer phrase.
- Use the passages and triples together.
- If one passage identifies a bridge entity and another passage gives the requested fact about that entity, combine them.
- supporting_passage_ids must come from the passage_id values shown above.
- Do not include explanation outside JSON.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def strip_code_fences(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = strip_code_fences(text)

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
            return None

    return None


def parse_answer_response(raw_response: str) -> Tuple[str, List[str], List[str], Optional[float], Dict[str, Any]]:
    obj = extract_json_object(raw_response)

    if obj is None:
        cleaned = strip_code_fences(raw_response).strip()
        cleaned = cleaned.split("\n")[0].strip()
        return cleaned, [], [], None, {}

    answer = (
        obj.get("answer")
        or obj.get("final_answer")
        or obj.get("predicted_answer")
        or ""
    )

    supporting_passage_ids = obj.get("supporting_passage_ids")
    if not isinstance(supporting_passage_ids, list):
        supporting_passage_ids = []

    supporting_triples = obj.get("supporting_triples")
    if not isinstance(supporting_triples, list):
        supporting_triples = []

    confidence = obj.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_value: Optional[float] = float(confidence)
    else:
        confidence_value = None

    return (
        str(answer).strip(),
        [str(x) for x in supporting_passage_ids],
        [str(x) for x in supporting_triples],
        confidence_value,
        obj,
    )


class MockAnswerBackend:
    def generate(self, messages: Sequence[Dict[str, str]], max_tokens: int = 256, **kwargs: Any) -> str:
        return json.dumps(
            {
                "answer": "I don't know",
                "supporting_passage_ids": [],
                "supporting_triples": [],
                "confidence": 0.0,
            }
        )


def build_answer_backend(args: argparse.Namespace) -> Any:
    if args.backend == "mock":
        return MockAnswerBackend()

    config = OpenIEExtractorConfig(
        backend=args.backend,
        model_name=args.model_name,
        temperature=args.temperature,
        max_ner_tokens=args.max_output_tokens,
        max_triple_tokens=args.max_output_tokens,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )

    return build_backend(config)


def generate_one_answer(
    backend: Any,
    messages: Sequence[Dict[str, str]],
    max_output_tokens: int,
) -> str:
    try:
        raw = backend.generate(
            messages=list(messages),
            max_tokens=max_output_tokens,
        )
        return normalize_generation_output(raw)
    except TypeError:
        # Fallback for backends that expect a prompt string.
        prompt = "\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}"
            for m in messages
        )
        raw = backend.generate(prompt, max_tokens=max_output_tokens)
        return normalize_generation_output(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate final answers from retrieval trajectories."
    )

    parser.add_argument("--retrieval_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "vllm", "transformers", "mock"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama70b-filter",
        help="Model name or served-model-name.",
    )

    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=256)

    # Only used for local vLLM/transformers backend.
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")

    parser.add_argument("--top_k_evidence", type=int, default=5)
    parser.add_argument("--top_k_filtered_triples", type=int, default=20)
    parser.add_argument("--max_passage_chars", type=int, default=2500)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output instead of resuming.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not os.path.exists(args.retrieval_path):
        raise FileNotFoundError(f"retrieval_path not found: {args.retrieval_path}")

    logger.info("Loading retrieval trajectories from %s", args.retrieval_path)
    rows = load_json_or_jsonl(args.retrieval_path)

    if args.start > 0:
        rows = rows[args.start :]

    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    logger.info("Loaded %d retrieval rows.", len(rows))

    output_rows: List[Dict[str, Any]] = []
    done_ids = set()

    if os.path.exists(args.output_path) and not args.force:
        output_rows = load_existing_rows(args.output_path)
        done_ids = existing_question_ids(output_rows)
        logger.info("Resume mode: found %d existing answers.", len(done_ids))

    logger.info("Loading answer backend: backend=%s model=%s", args.backend, args.model_name)
    backend = build_answer_backend(args)

    pending = []
    for idx, row in enumerate(rows):
        qid = str(safe_get(row, ["question_id", "id", "qid"], idx))

        if qid in done_ids:
            continue

        pending.append((idx, qid, row))

    logger.info("Pending answers: %d", len(pending))

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]

        for idx, qid, row in batch:
            source_row = row
            row = unwrap_trajectory_row(row)
            question = str(safe_get(row, ["question", "query"], ""))

            evidence_passages = collect_evidence_passages(
                row=row,
                top_k=args.top_k_evidence,
                max_passage_chars=args.max_passage_chars,
            )

            filtered_triples = collect_filtered_triples(
                row=row,
                top_k=args.top_k_filtered_triples,
            )

            messages = build_answer_messages(
                question=question,
                evidence_passages=evidence_passages,
                filtered_triples=filtered_triples,
            )

            logger.info(
                "Generating answer for %s with %d passages and %d filtered triples.",
                qid,
                len(evidence_passages),
                len(filtered_triples),
            )

            raw_response = generate_one_answer(
                backend=backend,
                messages=messages,
                max_output_tokens=args.max_output_tokens,
            )

            (
                predicted_answer,
                supporting_passage_ids,
                supporting_triples,
                confidence,
                parsed_response,
            ) = parse_answer_response(raw_response)

            answer_row = {
                "question_id": qid,
                "question": question,
                "predicted_answer": predicted_answer,
                "gold_answers": as_string_list(
                    safe_get(row, ["gold_answers", "answers", "answer"], [])
                ),
                "gold_passage_ids": as_string_list(
                    safe_get(row, ["gold_passage_ids", "supporting_passage_ids"], [])
                ),
                "controller_final_answer": (
                    row.get("controller_final_answer")
                    or row.get("final_answer")
                    or source_row.get("controller_final_answer")
                    ),
                "supporting_passage_ids": supporting_passage_ids,
                "supporting_triples": supporting_triples,
                "confidence": confidence,
                "evidence_passages": evidence_passages,
                "filtered_triples": filtered_triples,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
                "metadata": {
                    "answer_model": args.model_name,
                    "answer_backend": args.backend,
                    "retrieval_path": args.retrieval_path,
                    "top_k_evidence": args.top_k_evidence,
                    "top_k_filtered_triples": args.top_k_filtered_triples,
                    "source_format": "grpo_rollout" if isinstance(source_row.get("trajectory"), dict) else "retrieval_trajectory",
                },
            }

            output_rows.append(answer_row)

        # Save after every batch.
        save_jsonl(args.output_path, output_rows)

    logger.info("Saved %d answers to %s", len(output_rows), args.output_path)

    print("\nAnswer generation finished.")
    print(f"num_answers: {len(output_rows)}")
    print(f"output_path: {args.output_path}")


if __name__ == "__main__":
    main()