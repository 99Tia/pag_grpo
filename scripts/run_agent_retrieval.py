from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
SCRIPT_DIR = os.path.dirname(__file__)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


from build_triple_index import build_embedding_model  # noqa: E402
from ppr_agent.agent_env import (  # noqa: E402
    AgentEnv,
    AgentEnvConfig,
    save_trajectories_jsonl,
)
from ppr_agent.ppr_search import PPRSearchConfig, PPRSearchEngine  # noqa: E402
from ppr_agent.evidence_selector import (  # noqa: E402
    EvidenceSelectorConfig,
    EvidenceSelectorV2,
)
from ppr_agent.evidence_fusion import (  # noqa: E402
    EvidenceFusionConfig,
    HybridEvidenceFuser,
)
from ppr_agent.answer_reader import (  # noqa: E402
    AnswerReaderConfig,
    GroundedAnswerReader,
)
from ppr_agent.reasoning_agent import ReasoningAgent, ReasoningAgentConfig  # noqa: E402
from ppr_agent.schema import compute_mdhash_id  # noqa: E402
from ppr_agent.triple_filter import TripleFilterConfig  # noqa: E402

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


def unwrap_examples(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "examples", "queries", "questions"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    raise ValueError(
        "Unsupported question file format. Expected list or dict with "
        "data/examples/queries/questions."
    )


def get_first_available(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def normalize_answer_field(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [str(v) for v in value]

    return [str(value)]


def infer_gold_passage_ids(example: Dict[str, Any]) -> List[str]:
    direct = get_first_available(
        example,
        keys=[
            "gold_passage_ids",
            "supporting_passage_ids",
            "supporting_ids",
            "support_idxs",
        ],
    )

    if isinstance(direct, list):
        return [str(x) for x in direct]

    gold_ids: List[str] = []

    paragraphs = example.get("paragraphs")
    if isinstance(paragraphs, list):
        for para in paragraphs:
            if not isinstance(para, dict):
                continue

            is_supporting = bool(
                para.get("is_supporting")
                or para.get("supporting")
                or para.get("is_gold")
            )

            if not is_supporting:
                continue

            pid = get_first_available(
                para,
                keys=["idx", "id", "passage_id", "chunk_id"],
            )

            if pid is not None:
                gold_ids.append(str(pid))
                continue

            text = get_first_available(
                para,
                keys=["text", "content", "passage", "paragraph"],
            )
            if text is not None:
                gold_ids.append(compute_mdhash_id(str(text), prefix="chunk-"))

    return unique_preserve_order(gold_ids)


def unique_preserve_order(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()

    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)

    return output


def load_question_examples(
    path: str,
    start: int = 0,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    payload = load_json_or_jsonl(path)
    rows = unwrap_examples(payload)

    if start > 0:
        rows = rows[start:]

    if limit is not None and limit > 0:
        rows = rows[:limit]

    examples: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        question = get_first_available(row, keys=["question", "query"])
        if question is None:
            continue

        qid = get_first_available(row, keys=["id", "qid", "question_id"])
        if qid is None:
            qid = str(i)

        answers = get_first_available(row, keys=["answer", "answers", "gold_answers"])
        gold_answers = normalize_answer_field(answers)

        examples.append(
            {
                "id": str(qid),
                "question": str(question),
                "answer": gold_answers,
                "gold_passage_ids": infer_gold_passage_ids(row),
                "raw_example": row,
            }
        )

    if not examples:
        raise ValueError(f"No question examples loaded from {path}")

    return examples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run iterative PPR-agent retrieval."
    )
    parser.add_argument("--questions_path", type=str, required=True)
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)

    # Embedding model for query encoding
    parser.add_argument(
        "--embedding_backend",
        type=str,
        default="hf",
        choices=["mock", "hf", "sentence_transformers"],
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="nvidia/NV-Embed-v2",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--mock_dim", type=int, default=128)

    # SearchGraph/PPR settings
    parser.add_argument("--top_k_candidate_triples", type=int, default=50)
    parser.add_argument("--top_k_passages", type=int, default=5)
    parser.add_argument("--dense_reset_top_k", type=int, default=50)
    parser.add_argument("--linking_top_k", type=int, default=50)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.5)

    # Triple filter settings
    parser.add_argument("--enable_triple_filter", action="store_true")
    parser.add_argument(
        "--triple_filter_backend",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "transformers", "mock"],
    )
    parser.add_argument(
        "--triple_filter_model_name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument("--triple_filter_temperature", type=float, default=0.0)
    parser.add_argument("--triple_filter_max_tokens", type=int, default=512)
    parser.add_argument("--triple_filter_tensor_parallel_size", type=int, default=2)
    parser.add_argument("--triple_filter_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--triple_filter_base_url", type=str, default=None)
    parser.add_argument("--triple_filter_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--dspy_prompt_path", type=str, default=None)

    # Reasoning agent settings
    parser.add_argument(
        "--reasoning_backend",
        type=str,
        default="mock",
        choices=["vllm", "openai", "transformers", "mock"],
    )
    parser.add_argument(
        "--reasoning_model_name",
        type=str,
        default="/home/ib5539/models/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--reasoning_temperature", type=float, default=0.0)
    parser.add_argument("--reasoning_max_tokens", type=int, default=512)
    parser.add_argument("--reasoning_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--reasoning_gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--reasoning_base_url", type=str, default=None)
    parser.add_argument("--reasoning_api_key_env", type=str, default="OPENAI_API_KEY")

    # Agent loop
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_search_calls", type=int, default=4)
    parser.add_argument("--max_evidence_passages", type=int, default=20)
    parser.add_argument("--fallback_answer", type=str, default="I don't know")

    # End-of-trajectory finalization:
    # selector v2 -> hybrid keep2 -> frozen answer reader
    parser.add_argument(
        "--enable_finalization",
        action="store_true",
        help=(
            "Run EvidenceSelectorV2, hybrid PPR-LLM fusion, and the "
            "grounded answer reader once after each complete trajectory."
        ),
    )

    # Chain-aware evidence selector v2
    parser.add_argument(
        "--selector_model_name",
        type=str,
        default="llama70b-filter",
    )
    parser.add_argument(
        "--selector_base_url",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible selector endpoint. If omitted, "
            "--triple_filter_base_url is used."
        ),
    )
    parser.add_argument(
        "--selector_api_key_env",
        type=str,
        default="OPENAI_API_KEY",
    )
    parser.add_argument("--selector_top_pool", type=int, default=15)
    parser.add_argument("--selector_select_k", type=int, default=5)
    parser.add_argument(
        "--selector_max_passage_chars",
        type=int,
        default=900,
    )
    parser.add_argument("--selector_max_triples", type=int, default=30)
    parser.add_argument("--selector_temperature", type=float, default=0.0)
    parser.add_argument("--selector_max_tokens", type=int, default=700)
    parser.add_argument("--selector_retries", type=int, default=3)

    # Hybrid PPR + LLM-selected evidence fusion
    parser.add_argument(
        "--fusion_keep_ppr_top_n",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--fusion_target_top_k",
        type=int,
        default=5,
    )

    # Frozen grounded answer reader
    parser.add_argument(
        "--answer_backend",
        type=str,
        default="openai",
        choices=["openai", "vllm", "transformers", "mock"],
    )
    parser.add_argument(
        "--answer_model_name",
        type=str,
        default="llama70b-filter",
    )
    parser.add_argument(
        "--answer_base_url",
        type=str,
        default=None,
        help=(
            "Answer-reader endpoint. If omitted, --selector_base_url "
            "or --triple_filter_base_url is used."
        ),
    )
    parser.add_argument(
        "--answer_api_key_env",
        type=str,
        default="OPENAI_API_KEY",
    )
    parser.add_argument("--answer_temperature", type=float, default=0.0)
    parser.add_argument(
        "--answer_max_output_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--answer_tensor_parallel_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--answer_gpu_memory_utilization",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--answer_device_map",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--answer_top_k_evidence",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--answer_top_k_filtered_triples",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--answer_max_passage_chars",
        type=int,
        default=2500,
    )

    # Debug
    parser.add_argument("--save_step_debug", action="store_true")
    parser.add_argument("--debug_dir", type=str, default=None)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    logger.info("Loading questions from %s", args.questions_path)
    examples = load_question_examples(
        path=args.questions_path,
        start=args.start,
        limit=args.limit,
    )
    logger.info("Loaded %d examples.", len(examples))

    logger.info("Building embedding model for query encoding.")
    embedding_model = build_embedding_model(args)

    triple_filter_config = None
    if args.enable_triple_filter:
        triple_filter_config = TripleFilterConfig(
            backend=args.triple_filter_backend,
            model_name=args.triple_filter_model_name,
            dspy_prompt_path=args.dspy_prompt_path,
            enabled=True,
            fallback_to_input_if_empty=True,
            max_output_triples=None,
            temperature=args.triple_filter_temperature,
            max_tokens=args.triple_filter_max_tokens,
            base_url=args.triple_filter_base_url,
            api_key_env=args.triple_filter_api_key_env,
            tensor_parallel_size=args.triple_filter_tensor_parallel_size,
            gpu_memory_utilization=args.triple_filter_gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
        )

    search_config = PPRSearchConfig(
        graph_dir=args.graph_dir,
        index_dir=args.index_dir,
        top_k_candidate_triples=args.top_k_candidate_triples,
        enable_triple_filter=args.enable_triple_filter,
        fallback_to_candidates_if_filter_empty=True,
        max_filtered_triples=None,
        linking_top_k=args.linking_top_k,
        passage_node_weight=args.passage_node_weight,
        dense_reset_top_k=args.dense_reset_top_k,
        damping=args.damping,
        top_k_passages=args.top_k_passages,
        save_debug=args.save_step_debug,
        debug_dir=args.debug_dir,
    )

    logger.info("Loading PPR SearchGraph engine.")
    search_engine = PPRSearchEngine(
        config=search_config,
        embedding_model=embedding_model,
        triple_filter_config=triple_filter_config,
    )

    reasoning_config = ReasoningAgentConfig(
        backend=args.reasoning_backend,
        model_name=args.reasoning_model_name,
        temperature=args.reasoning_temperature,
        max_tokens=args.reasoning_max_tokens,
        max_search_steps=args.max_search_calls,
        default_top_k_triples=args.top_k_candidate_triples,
        default_top_k_passages=args.top_k_passages,
        tensor_parallel_size=args.reasoning_tensor_parallel_size,
        gpu_memory_utilization=args.reasoning_gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        base_url=args.reasoning_base_url,
        api_key_env=args.reasoning_api_key_env,
    )

    logger.info("Loading reasoning agent: backend=%s", args.reasoning_backend)
    reasoning_agent = ReasoningAgent(reasoning_config)

    evidence_selector = None
    evidence_fuser = None
    answer_reader = None

    if args.enable_finalization:
        selector_base_url = (
            args.selector_base_url
            or args.triple_filter_base_url
        )

        answer_base_url = (
            args.answer_base_url
            or selector_base_url
            or args.triple_filter_base_url
        )

        if not selector_base_url:
            raise ValueError(
                "Finalization is enabled, but no selector endpoint was "
                "provided. Set --selector_base_url or "
                "--triple_filter_base_url."
            )

        if args.answer_backend == "openai" and not answer_base_url:
            raise ValueError(
                "answer_backend=openai requires --answer_base_url, "
                "--selector_base_url, or --triple_filter_base_url."
            )

        logger.info(
            "Loading chain-aware evidence selector: model=%s endpoint=%s",
            args.selector_model_name,
            selector_base_url,
        )
        evidence_selector = EvidenceSelectorV2(
            EvidenceSelectorConfig(
                base_url=selector_base_url,
                model_name=args.selector_model_name,
                api_key_env=args.selector_api_key_env,
                top_pool=args.selector_top_pool,
                select_k=args.selector_select_k,
                max_passage_chars=args.selector_max_passage_chars,
                max_triples=args.selector_max_triples,
                temperature=args.selector_temperature,
                max_tokens=args.selector_max_tokens,
                retries=args.selector_retries,
                fallback_to_original_order=True,
            )
        )

        logger.info(
            "Configuring hybrid evidence fusion: keep_ppr_top_n=%d "
            "target_top_k=%d",
            args.fusion_keep_ppr_top_n,
            args.fusion_target_top_k,
        )
        evidence_fuser = HybridEvidenceFuser(
            EvidenceFusionConfig(
                keep_ppr_top_n=args.fusion_keep_ppr_top_n,
                target_top_k=args.fusion_target_top_k,
                copy_passages=True,
            )
        )

        logger.info(
            "Loading grounded answer reader: backend=%s model=%s endpoint=%s",
            args.answer_backend,
            args.answer_model_name,
            answer_base_url,
        )
        answer_reader = GroundedAnswerReader(
            AnswerReaderConfig(
                backend=args.answer_backend,
                model_name=args.answer_model_name,
                base_url=answer_base_url,
                api_key_env=args.answer_api_key_env,
                temperature=args.answer_temperature,
                max_output_tokens=args.answer_max_output_tokens,
                tensor_parallel_size=args.answer_tensor_parallel_size,
                gpu_memory_utilization=(
                    args.answer_gpu_memory_utilization
                ),
                trust_remote_code=args.trust_remote_code,
                device_map=args.answer_device_map,
                top_k_evidence=args.answer_top_k_evidence,
                top_k_filtered_triples=(
                    args.answer_top_k_filtered_triples
                ),
                max_passage_chars=args.answer_max_passage_chars,
                validate_support_ids=True,
            )
        )

    env = AgentEnv(
        config=AgentEnvConfig(
            max_steps=args.max_steps,
            max_search_calls=args.max_search_calls,
            max_evidence_passages=args.max_evidence_passages,
            fallback_answer=args.fallback_answer,
            enable_finalization=args.enable_finalization,
            preserve_base_evidence=True,
        ),
        reasoning_agent=reasoning_agent,
        search_engine=search_engine,
        evidence_selector=evidence_selector,
        evidence_fuser=evidence_fuser,
        answer_reader=answer_reader,
    )

    trajectories = env.run_batch(
        examples=examples,
        question_key="question",
        id_key="id",
        answers_key="answer",
        gold_passage_ids_key="gold_passage_ids",
    )

    save_trajectories_jsonl(args.output_path, trajectories)

    logger.info("Saved %d trajectories to %s", len(trajectories), args.output_path)

    # Print a tiny summary.
    num_search_calls = [
        traj.metadata.get("num_search_calls", traj.num_search_calls())
        for traj in trajectories
    ]
    avg_search_calls = sum(num_search_calls) / max(len(num_search_calls), 1)

    print("\nAgent retrieval finished.")
    print(f"num_examples: {len(trajectories)}")
    print(f"avg_search_calls: {avg_search_calls:.4f}")
    print(f"finalization_enabled: {args.enable_finalization}")

    if args.enable_finalization:
        predicted_answers = [
            traj.metadata.get("reader_predicted_answer")
            for traj in trajectories
        ]
        num_reader_answers = sum(
            answer is not None
            for answer in predicted_answers
        )
        print(f"num_reader_answers: {num_reader_answers}")
        print(
            "fusion_setting: "
            f"keep{args.fusion_keep_ppr_top_n}, "
            f"top_k={args.fusion_target_top_k}"
        )

    print(f"output_path: {args.output_path}")


if __name__ == "__main__":
    main()