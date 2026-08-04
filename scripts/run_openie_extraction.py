from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from ppr_agent.openie_extractor import (  # noqa: E402
    OpenIEExtractor,
    OpenIEExtractorConfig,
    save_openie_results,
)
from ppr_agent.schema import Passage  # noqa: E402


logger = logging.getLogger(__name__)


def load_json_or_jsonl(path: str) -> Any:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "docs", "corpus", "passages", "examples"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    raise ValueError(
        "Unsupported input format. Expected a list or a dict with one of: "
        "data, docs, corpus, passages, examples."
    )


def get_first_available(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def row_to_passage(row: Dict[str, Any], fallback_idx: int) -> Optional[Passage]:
    text = get_first_available(
        row,
        keys=["text", "content", "passage", "paragraph", "body"],
    )

    if text is None:
        return None

    text = str(text).strip()
    if not text:
        return None

    passage_id = get_first_available(
        row,
        keys=["idx", "id", "passage_id", "chunk_id"],
    )

    if passage_id is not None:
        passage_id = str(passage_id)

    title = get_first_available(row, keys=["title", "name"])
    if title is not None:
        title = str(title)

    source_id = get_first_available(row, keys=["source_id", "doc_id", "document_id"])
    if source_id is not None:
        source_id = str(source_id)

    metadata = {
        k: v
        for k, v in row.items()
        if k not in {"text", "content", "passage", "paragraph", "body"}
    }
    metadata["fallback_idx"] = fallback_idx

    return Passage.from_text(
        text=text,
        title=title,
        source_id=source_id,
        metadata=metadata,
        passage_id=passage_id,
    )


def flatten_examples_with_paragraphs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    seen = set()

    for ex in rows:
        paragraphs = ex.get("paragraphs")

        if isinstance(paragraphs, list):
            for para in paragraphs:
                if not isinstance(para, dict):
                    continue

                text = get_first_available(
                    para,
                    keys=["text", "content", "passage", "paragraph"],
                )
                if text is None:
                    continue

                pid = get_first_available(para, keys=["idx", "id", "passage_id", "chunk_id"])
                key = str(pid) if pid is not None else str(text)

                if key in seen:
                    continue

                seen.add(key)
                flattened.append(para)

        else:
            text = get_first_available(
                ex,
                keys=["text", "content", "passage", "paragraph"],
            )
            if text is None:
                continue

            pid = get_first_available(ex, keys=["idx", "id", "passage_id", "chunk_id"])
            key = str(pid) if pid is not None else str(text)

            if key in seen:
                continue

            seen.add(key)
            flattened.append(ex)

    return flattened


def load_passages(
    input_path: str,
    start: int = 0,
    limit: Optional[int] = None,
    flatten_paragraphs: bool = False,
) -> List[Passage]:
    payload = load_json_or_jsonl(input_path)
    rows = unwrap_payload(payload)

    if flatten_paragraphs:
        rows = flatten_examples_with_paragraphs(rows)

    if start > 0:
        rows = rows[start:]

    if limit is not None and limit > 0:
        rows = rows[:limit]

    passages: List[Passage] = []

    for idx, row in enumerate(tqdm(rows, desc="Loading passages")):
        if not isinstance(row, dict):
            continue

        passage = row_to_passage(row, fallback_idx=idx)
        if passage is not None:
            passages.append(passage)

    if not passages:
        raise ValueError(f"No passages found in {input_path}")

    return passages


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenIE extraction for PPR-agent framework."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input corpus JSON/JSONL path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output OpenIE JSON path.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "transformers", "mock"],
        help="LLM backend.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name or local model path.",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_ner_tokens", type=int, default=512)
    parser.add_argument("--max_triple_tokens", type=int, default=2048)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--flatten_paragraphs",
        action="store_true",
        help="Use this if input is a QA file where passages are inside examples[*].paragraphs.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing output and re-extract.",
    )

    # OpenAI / OpenAI-compatible
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--base_url", type=str, default=None)

    # vLLM
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--trust_remote_code", action="store_true")

    # transformers
    parser.add_argument("--device_map", type=str, default="auto")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    logger.info("Loading passages from %s", args.input_path)
    passages = load_passages(
        input_path=args.input_path,
        start=args.start,
        limit=args.limit,
        flatten_paragraphs=args.flatten_paragraphs,
    )

    logger.info("Loaded %d passages.", len(passages))
    logger.info("Backend: %s", args.backend)
    logger.info("Model: %s", args.model_name)

    config = OpenIEExtractorConfig(
        backend=args.backend,
        model_name=args.model_name,
        temperature=args.temperature,
        max_ner_tokens=args.max_ner_tokens,
        max_triple_tokens=args.max_triple_tokens,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        batch_size=args.batch_size,
        save_every=args.save_every,
        force=args.force,
    )

    extractor = OpenIEExtractor(config)

    docs = extractor.batch_extract(
        passages=passages,
        output_path=args.output_path,
    )

    save_openie_results(args.output_path, docs)

    num_entities = sum(len(doc.extracted_entities) for doc in docs)
    num_triples = sum(len(doc.extracted_triples) for doc in docs)

    logger.info("OpenIE extraction finished.")
    logger.info("Docs: %d", len(docs))
    logger.info("Entities: %d", num_entities)
    logger.info("Triples: %d", num_triples)
    logger.info("Saved to: %s", args.output_path)


if __name__ == "__main__":
    main()