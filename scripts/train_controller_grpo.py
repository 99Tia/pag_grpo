from __future__ import annotations
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_scheduler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (str(SRC_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
from build_triple_index import build_embedding_model  # noqa: E402
from ppr_agent.agent_env import AgentEnv, AgentEnvConfig  # noqa: E402
from ppr_agent.answer_reader import (  # noqa: E402
    AnswerReaderConfig,
    GroundedAnswerReader,
)
from ppr_agent.evidence_fusion import (  # noqa: E402
    EvidenceFusionConfig,
    HybridEvidenceFuser,
)
from ppr_agent.evidence_selector import (  # noqa: E402
    EvidenceSelectorConfig,
    EvidenceSelectorV2,
)
from ppr_agent.grpo_loss import (  # noqa: E402
    GRPOLoss,
    GRPOLossCollator,
    GRPOLossConfig,
)
from ppr_agent.grpo_policy import (  # noqa: E402
    GRPOPolicy,
    GRPOPolicyConfig,
)
from ppr_agent.grpo_rollout import (  # noqa: E402
    GRPORolloutCollector,
    GRPORolloutConfig,
    GRPORolloutExample,
    collect_policy_step_samples,
)
from ppr_agent.grpo_types import PolicyStepSample, RolloutGroup  # noqa: E402
from ppr_agent.ppr_search import PPRSearchConfig, PPRSearchEngine  # noqa: E402
from ppr_agent.reasoning_agent import (  # noqa: E402
    ReasoningAgent,
    ReasoningAgentConfig,
)
from ppr_agent.trajectory_reward import (  # noqa: E402
    TrajectoryRewardCalculator,
    TrajectoryRewardConfig,
)
from ppr_agent.triple_filter import TripleFilterConfig  # noqa: E402

logger = logging.getLogger("train_controller_grpo")


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {log_level!r}.")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose one of {sorted(aliases)}."
        )
    return aliases[normalized]


def parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]

def ensure_json_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): ensure_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_json_serializable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): ensure_json_serializable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)

def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ensure_json_serializable(record), ensure_ascii=False))
        handle.write("\n")
        handle.flush()

def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ensure_json_serializable(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_json_or_jsonl(path: str) -> Any:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Question file does not exist: {source}")

    if source.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {source}:{line_number}: {exc}"
                    ) from exc
                rows.append(row)
        return rows

    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def unwrap_examples(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]

    if isinstance(payload, Mapping):
        for key in ("data", "examples", "queries", "questions"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, Mapping)]

    raise ValueError(
        "Unsupported question format. Expected a list or a dictionary containing "
        "data/examples/queries/questions."
    )

def load_rollout_examples(
    path: str,
    *,
    start: int = 0,
    limit: Optional[int] = None,
) -> List[GRPORolloutExample]:
    rows = unwrap_examples(load_json_or_jsonl(path))
    if start < 0:
        raise ValueError("start must be non-negative.")
    rows = rows[start:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive or omitted.")
        rows = rows[:limit]

    examples: List[GRPORolloutExample] = []
    for index, row in enumerate(rows):
        try:
            example = GRPORolloutExample.from_mapping(row)
        except Exception as exc:
            raise ValueError(
                f"Could not convert question row {start + index} into a GRPO example: {exc}"
            ) from exc
        examples.append(example)

    if not examples:
        raise ValueError(f"No usable examples were loaded from {path}.")
    return examples


def deterministic_epoch_order(
    num_examples: int,
    *,
    epoch_index: int,
    seed: int,
    shuffle: bool,
) -> List[int]:
    order = list(range(num_examples))
    if shuffle:
        random.Random(seed + epoch_index).shuffle(order)
    return order

def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def cuda_memory_metrics(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    gib = float(1024**3)
    return {
        "cuda_allocated_gib": torch.cuda.memory_allocated(index) / gib,
        "cuda_reserved_gib": torch.cuda.memory_reserved(index) / gib,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(index) / gib,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(index) / gib,
    }


def reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(index)


# Model and LoRA construction
def _load_base_causal_lm(
    *,
    model_name_or_path: str,
    revision: str,
    dtype: torch.dtype,
    attn_implementation: Optional[str],
    trust_remote_code: bool,
) -> nn.Module:
    kwargs: Dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation and attn_implementation.lower() != "none":
        kwargs["attn_implementation"] = attn_implementation

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            **kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            **kwargs,
        )


def resolve_resume_checkpoint(value: Optional[str], output_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    if value != "latest":
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path

    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(
            f"--resume_from_checkpoint latest was requested, but no checkpoint-* "
            f"directory exists under {output_dir}."
        )
    return max(candidates, key=lambda item: item[0])[1]


def build_trainable_policy(
    args: argparse.Namespace,
    *,
    resume_checkpoint: Optional[Path],
) -> Tuple[GRPOPolicy, PreTrainedTokenizerBase]:
    dtype = resolve_dtype(args.torch_dtype)
    policy_device = torch.device(args.policy_device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Controller tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading base controller model from %s", args.model_name_or_path)
    base_model = _load_base_causal_lm(
        model_name_or_path=args.model_name_or_path,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    if resume_checkpoint is not None:
        adapter_config = resume_checkpoint / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"Resume directory has no adapter_config.json: {resume_checkpoint}"
            )
        logger.info("Loading trainable LoRA adapter from %s", resume_checkpoint)
        model = PeftModel.from_pretrained(
            base_model,
            str(resume_checkpoint),
            is_trainable=True,
        )
    else:
        target_modules = parse_csv(args.lora_target_modules)
        if not target_modules:
            raise ValueError("At least one LoRA target module is required.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias=args.lora_bias,
        )
        model = get_peft_model(base_model, lora_config)

    model.to(policy_device)

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            enable_input_require_grads()

    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    trainable, total = count_trainable_parameters(model)
    logger.info(
        "Controller parameters: trainable=%s total=%s trainable_ratio=%.6f%%",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )
    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters were found.")

    prompt_agent = ReasoningAgent(
        ReasoningAgentConfig(
            backend="mock",
            model_name=args.model_name_or_path,
            temperature=args.rollout_temperature,
            max_tokens=args.max_action_tokens,
            max_search_steps=args.max_search_calls,
            default_top_k_triples=args.top_k_candidate_triples,
            default_top_k_passages=args.top_k_passages,
            max_evidence_passages_in_prompt=args.max_evidence_passages_in_prompt,
            max_filtered_triples_in_prompt=args.max_filtered_triples_in_prompt,
            max_candidate_triples_in_prompt=args.max_candidate_triples_in_prompt,
            max_chars_per_passage=args.max_chars_per_passage_in_prompt,
            max_memory_text_chars=args.max_memory_text_chars,
        )
    )

    policy = GRPOPolicy(
        model=model,
        tokenizer=tokenizer,
        reasoning_agent=prompt_agent,
        config=GRPOPolicyConfig(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            model_revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            device=args.policy_device,
            device_map=None,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_action_tokens,
            min_new_tokens=args.min_action_tokens,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            repetition_penalty=args.rollout_repetition_penalty,
            do_sample=True,
            use_cache=True,
            include_eos_in_action_mask=True,
            store_full_prompt_text_in_metadata=args.store_full_prompt_text,
        ),
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    return policy, tokenizer


# Frozen retrieval
def build_frozen_environment(
    args: argparse.Namespace,
    *,
    prompt_agent: ReasoningAgent,
) -> AgentEnv:
    embedding_args = SimpleNamespace(
        embedding_backend=args.embedding_backend,
        embedding_model_name=args.embedding_model_name,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
        max_length=args.embedding_max_length,
        trust_remote_code=args.trust_remote_code,
        mock_dim=args.embedding_mock_dim,
        embedding_base_url=args.embedding_base_url,
        embedding_timeout=args.embedding_timeout,
    )

    logger.info(
        "Loading embedding backend=%s model=%s device=%s base_url=%s",
        args.embedding_backend,
        args.embedding_model_name,
        args.embedding_device,
        args.embedding_base_url if args.embedding_backend == "remote" else None,
    )
    embedding_model = build_embedding_model(embedding_args)

    triple_filter_config: Optional[TripleFilterConfig] = None
    if args.enable_triple_filter:
        if args.triple_filter_backend == "openai" and not args.triple_filter_base_url:
            raise ValueError(
                "Triple filtering through the frozen 70B server requires "
                "--triple_filter_base_url."
            )
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

    search_engine = PPRSearchEngine(
        config=PPRSearchConfig(
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
        ),
        embedding_model=embedding_model,
        triple_filter_config=triple_filter_config,
    )

    evidence_selector = None
    evidence_fuser = None
    answer_reader = None

    if args.enable_finalization:
        selector_base_url = args.selector_base_url or args.triple_filter_base_url
        answer_base_url = (
            args.answer_base_url
            or selector_base_url
            or args.triple_filter_base_url
        )
        if not selector_base_url:
            raise ValueError(
                "Finalization requires --selector_base_url or "
                "--triple_filter_base_url."
            )
        if args.answer_backend == "openai" and not answer_base_url:
            raise ValueError("answer_backend=openai requires an answer endpoint.")

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

        evidence_fuser = HybridEvidenceFuser(
            EvidenceFusionConfig(
                keep_ppr_top_n=args.fusion_keep_ppr_top_n,
                target_top_k=args.fusion_target_top_k,
                copy_passages=True,
            )
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
                gpu_memory_utilization=args.answer_gpu_memory_utilization,
                trust_remote_code=args.trust_remote_code,
                device_map=args.answer_device_map,
                top_k_evidence=args.answer_top_k_evidence,
                top_k_filtered_triples=args.answer_top_k_filtered_triples,
                max_passage_chars=args.answer_max_passage_chars,
                validate_support_ids=True,
            )
        )

    return AgentEnv(
        config=AgentEnvConfig(
            max_steps=args.max_steps,
            max_search_calls=args.max_search_calls,
            deduplicate_evidence=True,
            max_evidence_passages=args.max_evidence_passages,
            max_evidence_triples=args.max_evidence_triples,
            max_memory_text_chars=args.max_memory_text_chars,
            fallback_answer=args.fallback_answer,
            enable_finalization=args.enable_finalization,
            preserve_base_evidence=True,
            save_trajectories=False,
        ),
        reasoning_agent=prompt_agent,
        search_engine=search_engine,
        evidence_selector=evidence_selector,
        evidence_fuser=evidence_fuser,
        answer_reader=answer_reader,
    )



# Reward, rollout, and loss construction
def build_reward_calculator(args: argparse.Namespace) -> TrajectoryRewardCalculator:
    return TrajectoryRewardCalculator(
        TrajectoryRewardConfig(
            support_top_k=args.reward_support_top_k,
            max_search_calls=args.max_search_calls,
            allow_title_only_match=args.allow_title_only_support_match,
            title_only_requires_unique=True,
            answer_f1_weight=args.reward_answer_f1_weight,
            answer_exact_match_weight=args.reward_answer_em_weight,
            support_recall_weight=args.reward_support_recall_weight,
            support_precision_weight=args.reward_support_precision_weight,
            full_support_weight=args.reward_full_support_weight,
            format_validity_weight=args.reward_format_weight,
            evidence_novelty_weight=args.reward_novelty_weight,
            search_cost_weight=args.reward_search_cost_weight,
            duplicate_search_weight=args.reward_duplicate_search_weight,
            forced_stop_weight=args.reward_forced_stop_weight,
            unknown_answer_weight=args.reward_unknown_answer_weight,
            advantage_epsilon=args.advantage_epsilon,
            zero_variance_threshold=args.zero_variance_threshold,
        )
    )


def build_rollout_collector(
    args: argparse.Namespace,
    *,
    policy: GRPOPolicy,
    environment: AgentEnv,
    reward_calculator: TrajectoryRewardCalculator,
) -> GRPORolloutCollector:
    return GRPORolloutCollector(
        policy=policy,
        environment=environment,
        reward_calculator=reward_calculator,
        config=GRPORolloutConfig(
            group_size=args.group_size,
            base_seed=args.seed,
            sequential_groups=True,
            tokenize_observations=False,
            max_observation_passage_ids=10,
            score_trajectories=True,
            compute_advantages=True,
            raise_on_search_error=not args.continue_on_search_error,
            raise_on_finalization_error=not args.continue_on_finalization_error,
            validate_records=True,
        ),
    )


# Exact microbatched trajectory-mean GRPO update
def group_samples_by_trajectory(
    samples: Sequence[PolicyStepSample],
) -> "OrderedDict[str, List[PolicyStepSample]]":
    grouped: "OrderedDict[str, List[PolicyStepSample]]" = OrderedDict()
    for sample in samples:
        grouped.setdefault(sample.trajectory_id, []).append(sample)
    return grouped


def chunks(values: Sequence[PolicyStepSample], size: int) -> Iterable[List[PolicyStepSample]]:
    if size <= 0:
        raise ValueError("Microbatch size must be greater than zero.")
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def train_on_rollout_group(
    *,
    model: nn.Module,
    group: RolloutGroup,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    collator: GRPOLossCollator,
    loss_function: GRPOLoss,
    device: torch.device,
    max_policy_steps_per_microbatch: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    samples = collect_policy_step_samples([group])
    if not samples:
        raise ValueError(
            f"Rollout group {group.question_id!r} produced no policy-step samples."
        )

    by_trajectory = group_samples_by_trajectory(samples)
    num_trajectories = len(by_trajectory)
    if num_trajectories == 0:
        raise ValueError("No trajectories were available for optimization.")

    optimizer.zero_grad(set_to_none=True)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    weighted_metrics: Dict[str, float] = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl_loss": 0.0,
        "mean_reference_kl": 0.0,
        "mean_ratio": 0.0,
        "mean_log_ratio": 0.0,
        "clip_fraction": 0.0,
        "approximate_old_kl": 0.0,
    }
    num_microbatches = 0
    total_action_tokens = 0
    total_policy_steps = len(samples)

    for trajectory_id, trajectory_samples in by_trajectory.items():
        trajectory_token_count = sum(
            int(sum(sample.target_token_mask)) for sample in trajectory_samples
        )
        if trajectory_token_count <= 0:
            raise ValueError(
                f"Trajectory {trajectory_id!r} contains no active action tokens."
            )
        total_action_tokens += trajectory_token_count

        for micro_samples in chunks(
            trajectory_samples,
            max_policy_steps_per_microbatch,
        ):
            batch = collator(micro_samples).to(device)
            chunk_token_count = batch.num_action_tokens
            scale = (
                float(chunk_token_count)
                / float(trajectory_token_count)
                / float(num_trajectories)
            )

            output = loss_function(
                policy_model=model,
                batch=batch,
                reference_model=None,
                reference_log_probs=None,
                return_log_probs=False,
            )
            scaled_loss = output.loss * scale
            scaled_loss.backward()

            detached = output.detached_metrics()
            for key in weighted_metrics:
                weighted_metrics[key] += float(detached[key]) * scale

            num_microbatches += 1
            del output, scaled_loss, batch

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameter received a gradient.")

    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        max_norm=max_grad_norm,
    )
    grad_norm = float(grad_norm_tensor.detach().cpu())
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    weighted_metrics.update(
        {
            "grad_norm": grad_norm,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "num_microbatches": float(num_microbatches),
            "num_policy_steps": float(total_policy_steps),
            "num_action_tokens": float(total_action_tokens),
            "num_trajectories": float(num_trajectories),
        }
    )
    return weighted_metrics



# Diagnostics and checkpointing
def summarize_rollout_group(group: RolloutGroup) -> Dict[str, Any]:
    rewards = [float(trajectory.reward.total) for trajectory in group.trajectories]
    advantages = [float(trajectory.advantage or 0.0) for trajectory in group.trajectories]

    component_names = sorted(
        {
            name
            for trajectory in group.trajectories
            for name in trajectory.reward.components
        }
    )
    component_means: Dict[str, float] = {}
    for name in component_names:
        values = [
            float(trajectory.reward.components.get(name, 0.0))
            for trajectory in group.trajectories
        ]
        component_means[name] = float(sum(values) / max(len(values), 1))

    return {
        "question_id": group.question_id,
        "group_size": group.group_size,
        "reward_mean": float(group.reward_mean or 0.0),
        "reward_std": float(group.reward_std or 0.0),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "rewards": rewards,
        "advantages": advantages,
        "zero_variance": bool(group.zero_variance),
        "mean_num_steps": float(
            sum(t.num_steps for t in group.trajectories) / max(group.group_size, 1)
        ),
        "mean_search_calls": float(
            sum(t.num_search_calls for t in group.trajectories)
            / max(group.group_size, 1)
        ),
        "forced_stop_rate": float(
            sum(bool(t.forced_stop) for t in group.trajectories)
            / max(group.group_size, 1)
        ),
        "reward_component_means": component_means,
    }


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving checkpoint to %s", checkpoint_dir)

    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("Policy model does not expose save_pretrained().")
    save_pretrained(str(checkpoint_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(checkpoint_dir))

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_state": dict(state),
            "rng_state": capture_rng_state(),
        },
        checkpoint_dir / "training_state.pt",
    )
    write_json(checkpoint_dir / "training_args.json", vars(args))


def load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid training-state payload: {path}")
    return payload


def load_optimizer_state_if_available(
    *,
    resume_checkpoint: Optional[Path],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> Dict[str, Any]:
    if resume_checkpoint is None:
        return {
            "epoch_index": 0,
            "next_position_in_epoch": 0,
            "processed_groups": 0,
            "global_step": 0,
            "zero_variance_groups": 0,
        }

    state_path = resume_checkpoint / "training_state.pt"
    if not state_path.exists():
        logger.warning(
            "No training_state.pt in %s; adapter weights will resume, but "
            "optimizer/scheduler/data cursor will restart.",
            resume_checkpoint,
        )
        return {
            "epoch_index": 0,
            "next_position_in_epoch": 0,
            "processed_groups": 0,
            "global_step": 0,
            "zero_variance_groups": 0,
        }

    payload = load_torch_checkpoint(state_path)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])

    state = dict(payload.get("training_state") or {})
    logger.info(
        "Resumed optimizer state: global_step=%s processed_groups=%s epoch=%s position=%s",
        state.get("global_step", 0),
        state.get("processed_groups", 0),
        state.get("epoch_index", 0),
        state.get("next_position_in_epoch", 0),
    )
    return state


# Argument parser
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="True trajectory-level GRPO training for the PPR agent controller.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required project artifacts.
    parser.add_argument("--questions_path", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    # Dataset traversal.
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_groups", type=int, default=None)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Direct 8B policy + LoRA.
    parser.add_argument(
        "--model_name_or_path",
        default="/home/ib5539/models/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--model_revision", default="main")
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument(
        "--torch_dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager", "none"],
        default="sdpa",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora_bias", choices=["none", "all", "lora_only"], default="none")

    # On-policy controller generation.
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_top_p", type=float, default=1.0)
    parser.add_argument("--rollout_top_k", type=int, default=0)
    parser.add_argument("--rollout_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--max_action_tokens", type=int, default=512)
    parser.add_argument("--min_action_tokens", type=int, default=1)
    parser.add_argument("--store_full_prompt_text", action="store_true")

    # Prompt/evidence limits must match the validated baseline behavior.
    parser.add_argument("--max_evidence_passages_in_prompt", type=int, default=8)
    parser.add_argument("--max_filtered_triples_in_prompt", type=int, default=20)
    parser.add_argument("--max_candidate_triples_in_prompt", type=int, default=10)
    parser.add_argument("--max_chars_per_passage_in_prompt", type=int, default=900)
    parser.add_argument("--max_memory_text_chars", type=int, default=1400)

    # Optimization.
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument(
        "--lr_scheduler_type",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        default="cosine",
    )
    parser.add_argument("--num_policy_epochs", type=int, default=1)
    parser.add_argument("--max_policy_steps_per_microbatch", type=int, default=1)

    # Clipped GRPO objective.
    parser.add_argument("--epsilon_low", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument(
        "--apply_kl_when_advantage_is_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip_zero_variance_groups", action="store_true")

    # Reward.
    parser.add_argument("--reward_support_top_k", type=int, default=5)
    parser.add_argument("--reward_answer_f1_weight", type=float, default=2.0)
    parser.add_argument("--reward_answer_em_weight", type=float, default=0.0)
    parser.add_argument("--reward_support_recall_weight", type=float, default=1.0)
    parser.add_argument("--reward_support_precision_weight", type=float, default=0.0)
    parser.add_argument("--reward_full_support_weight", type=float, default=1.0)
    parser.add_argument("--reward_format_weight", type=float, default=0.10)
    parser.add_argument("--reward_novelty_weight", type=float, default=0.10)
    parser.add_argument("--reward_search_cost_weight", type=float, default=-0.05)
    parser.add_argument("--reward_duplicate_search_weight", type=float, default=-0.15)
    parser.add_argument("--reward_forced_stop_weight", type=float, default=-0.10)
    parser.add_argument("--reward_unknown_answer_weight", type=float, default=-0.10)
    parser.add_argument("--advantage_epsilon", type=float, default=1e-4)
    parser.add_argument("--zero_variance_threshold", type=float, default=1e-8)
    parser.add_argument(
        "--allow_title_only_support_match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # SearchGraph/PPR and evidence memory.
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_search_calls", type=int, default=4)
    parser.add_argument("--fallback_answer", default="I don't know")
    parser.add_argument("--max_evidence_passages", type=int, default=20)
    parser.add_argument("--max_evidence_triples", type=int, default=80)
    parser.add_argument("--top_k_candidate_triples", type=int, default=40)
    parser.add_argument("--top_k_passages", type=int, default=10)
    parser.add_argument("--dense_reset_top_k", type=int, default=50)
    parser.add_argument("--linking_top_k", type=int, default=50)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.5)

    parser.add_argument(
        "--embedding_backend",
        choices=["mock", "hf", "sentence_transformers", "remote"],
        default="remote",
    )
    parser.add_argument("--embedding_model_name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_device", default="cpu")
    parser.add_argument("--embedding_batch_size", type=int, default=2)
    parser.add_argument("--embedding_max_length", type=int, default=4096)
    parser.add_argument("--embedding_mock_dim", type=int, default=128)
    parser.add_argument(
        "--embedding_base_url",
        default="http://localhost:8003",
        help=(
            "Base URL of the remote embedding service when "
            "--embedding_backend=remote."
        ),
    )
    parser.add_argument(
        "--embedding_timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for one remote embedding request.",
    )

    # Frozen 70B triple filter.
    parser.add_argument(
        "--enable_triple_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--triple_filter_backend",
        choices=["openai", "vllm", "transformers", "mock"],
        default="openai",
    )
    parser.add_argument("--triple_filter_model_name", default="llama70b-filter")
    parser.add_argument("--triple_filter_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--triple_filter_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--triple_filter_temperature", type=float, default=0.0)
    parser.add_argument("--triple_filter_max_tokens", type=int, default=512)
    parser.add_argument("--triple_filter_tensor_parallel_size", type=int, default=2)
    parser.add_argument("--triple_filter_gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--dspy_prompt_path", default=None)

    # Frozen finalization: selector -> hybrid fusion -> grounded reader.
    parser.add_argument(
        "--enable_finalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--selector_model_name", default="llama70b-filter")
    parser.add_argument("--selector_base_url", default=None)
    parser.add_argument("--selector_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--selector_top_pool", type=int, default=15)
    parser.add_argument("--selector_select_k", type=int, default=5)
    parser.add_argument("--selector_max_passage_chars", type=int, default=900)
    parser.add_argument("--selector_max_triples", type=int, default=30)
    parser.add_argument("--selector_temperature", type=float, default=0.0)
    parser.add_argument("--selector_max_tokens", type=int, default=700)
    parser.add_argument("--selector_retries", type=int, default=3)

    parser.add_argument("--fusion_keep_ppr_top_n", type=int, default=2)
    parser.add_argument("--fusion_target_top_k", type=int, default=5)

    parser.add_argument(
        "--answer_backend",
        choices=["openai", "vllm", "transformers", "mock"],
        default="openai",
    )
    parser.add_argument("--answer_model_name", default="llama70b-filter")
    parser.add_argument("--answer_base_url", default=None)
    parser.add_argument("--answer_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--answer_temperature", type=float, default=0.0)
    parser.add_argument("--answer_max_output_tokens", type=int, default=256)
    parser.add_argument("--answer_tensor_parallel_size", type=int, default=2)
    parser.add_argument("--answer_gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--answer_device_map", default="auto")
    parser.add_argument("--answer_top_k_evidence", type=int, default=10)
    parser.add_argument("--answer_top_k_filtered_triples", type=int, default=30)
    parser.add_argument("--answer_max_passage_chars", type=int, default=2500)

    # Saving, resumption, and controlled testing.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--save_rollouts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout_only", action="store_true")
    parser.add_argument("--continue_on_search_error", action="store_true")
    parser.add_argument("--continue_on_finalization_error", action="store_true")
    parser.add_argument("--save_step_debug", action="store_true")
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--log_level", default="INFO")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "num_train_epochs",
        "group_size",
        "max_action_tokens",
        "max_prompt_tokens",
        "max_steps",
        "max_search_calls",
        "max_policy_steps_per_microbatch",
        "num_policy_epochs",
        "save_steps",
    )
    for name in positive_integer_fields:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be greater than zero.")

    if args.group_size <= 1:
        raise ValueError("--group_size must be greater than one for GRPO.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive or omitted.")
    if args.max_train_groups is not None and args.max_train_groups <= 0:
        raise ValueError("--max_train_groups must be positive or omitted.")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive.")
    if args.max_grad_norm <= 0:
        raise ValueError("--max_grad_norm must be positive.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0, 1).")
    if args.save_total_limit <= 0:
        raise ValueError("--save_total_limit must be greater than zero.")
    if args.embedding_timeout <= 0:
        raise ValueError("--embedding_timeout must be greater than zero.")
    if args.embedding_backend == "remote" and not str(args.embedding_base_url or "").strip():
        raise ValueError(
            "--embedding_base_url is required when --embedding_backend=remote."
        )

    policy_device = torch.device(args.policy_device)
    if policy_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA policy device was requested, but CUDA is unavailable.")
    if policy_device.type == "cuda":
        index = policy_device.index if policy_device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Policy device {args.policy_device} is not visible. "
                f"Visible CUDA device count: {torch.cuda.device_count()}."
            )


def rotate_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    checkpoints: List[Tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and suffix.isdigit():
            checkpoints.append((int(suffix), path))
    checkpoints.sort(key=lambda item: item[0])

    while len(checkpoints) > save_total_limit:
        _, path = checkpoints.pop(0)
        logger.info("Removing old checkpoint %s", path)
        import shutil

        shutil.rmtree(path)


# Main training loop
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    configure_logging(args.log_level)
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    rollout_path = output_dir / "rollouts.jsonl"
    write_json(output_dir / "training_args.json", vars(args))

    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        output_dir,
    )

    logger.info("Loading GRPO training questions from %s", args.questions_path)
    examples = load_rollout_examples(
        args.questions_path,
        start=args.start,
        limit=args.limit,
    )
    logger.info("Loaded %d training questions.", len(examples))

    policy, tokenizer = build_trainable_policy(
        args,
        resume_checkpoint=resume_checkpoint,
    )
    model = policy.model
    policy_device = policy.input_device

    environment = build_frozen_environment(
        args,
        prompt_agent=policy.reasoning_agent,
    )
    reward_calculator = build_reward_calculator(args)
    collector = build_rollout_collector(
        args,
        policy=policy,
        environment=environment,
        reward_calculator=reward_calculator,
    )

    planned_groups = len(examples) * args.num_train_epochs
    if args.max_train_groups is not None:
        planned_groups = min(planned_groups, args.max_train_groups)
    planned_optimizer_steps = max(
        1,
        planned_groups * args.num_policy_epochs,
    )
    warmup_steps = int(round(planned_optimizer_steps * args.warmup_ratio))

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_optimizer_steps,
    )

    training_state = load_optimizer_state_if_available(
        resume_checkpoint=resume_checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    epoch_start = int(training_state.get("epoch_index", 0))
    position_start = int(training_state.get("next_position_in_epoch", 0))
    processed_groups = int(training_state.get("processed_groups", 0))
    global_step = int(training_state.get("global_step", 0))
    zero_variance_groups = int(training_state.get("zero_variance_groups", 0))

    collator = GRPOLossCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        pad_to_multiple_of=8,
        validate_samples=True,
    )
   
    loss_function = GRPOLoss(
        GRPOLossConfig(
            epsilon_low=args.epsilon_low,
            epsilon_high=args.epsilon_high,
            beta=args.kl_beta,
            reduction="token_mean",
            use_reference_kl=args.kl_beta > 0.0,
            apply_kl_when_advantage_is_zero=(
                args.apply_kl_when_advantage_is_zero
            ),
            validate_trajectory_consistency=True,
        )
    )

    logger.info(
        "Starting GRPO: groups=%d group_size=%d policy_epochs=%d "
        "planned_optimizer_steps=%d policy_device=%s embedding_backend=%s "
        "embedding_device=%s embedding_base_url=%s",
        planned_groups,
        args.group_size,
        args.num_policy_epochs,
        planned_optimizer_steps,
        policy_device,
        args.embedding_backend,
        args.embedding_device,
        args.embedding_base_url if args.embedding_backend == "remote" else None,
    )

    training_started = time.perf_counter()
    stop_training = False
    cursor_state: Dict[str, Any] = {
        "epoch_index": epoch_start,
        "next_position_in_epoch": position_start,
        "processed_groups": processed_groups,
        "global_step": global_step,
        "zero_variance_groups": zero_variance_groups,
        "elapsed_seconds": 0.0,
    }

    for epoch_index in range(epoch_start, args.num_train_epochs):
        order = deterministic_epoch_order(
            len(examples),
            epoch_index=epoch_index,
            seed=args.seed,
            shuffle=args.shuffle,
        )
        first_position = position_start if epoch_index == epoch_start else 0

        for position_in_epoch in range(first_position, len(order)):
            if (
                args.max_train_groups is not None
                and processed_groups >= args.max_train_groups
            ):
                stop_training = True
                break

            example_index = order[position_in_epoch]
            example = examples[example_index]
            rollout_iteration = processed_groups

            reset_cuda_peak_memory(policy_device)
            rollout_started = time.perf_counter()
            logger.info(
                "Collecting rollout group %d | epoch=%d position=%d qid=%s",
                processed_groups,
                epoch_index,
                position_in_epoch,
                example.question_id,
            )

            group = collector.collect_group(
                example,
                rollout_iteration=rollout_iteration,
            )
            rollout_seconds = time.perf_counter() - rollout_started
            group_summary = summarize_rollout_group(group)
            if group.zero_variance:
                zero_variance_groups += 1

            if args.save_rollouts:
                append_jsonl(rollout_path, group.to_dict())

            rollout_record: Dict[str, Any] = {
                "event": "rollout_group",
                "epoch": epoch_index,
                "position_in_epoch": position_in_epoch,
                "processed_groups": processed_groups,
                "global_step": global_step,
                "rollout_seconds": rollout_seconds,
                "zero_variance_rate_so_far": (
                    zero_variance_groups / float(processed_groups + 1)
                ),
                **group_summary,
                **cuda_memory_metrics(policy_device),
            }
            append_jsonl(metrics_path, rollout_record)
            logger.info(
                "Rollout qid=%s reward=%.4f±%.4f zero_variance=%s "
                "steps=%.2f searches=%.2f seconds=%.1f",
                group.question_id,
                float(group.reward_mean or 0.0),
                float(group.reward_std or 0.0),
                bool(group.zero_variance),
                group_summary["mean_num_steps"],
                group_summary["mean_search_calls"],
                rollout_seconds,
            )

            should_skip = bool(
                group.zero_variance and args.skip_zero_variance_groups
            )
            if not args.rollout_only:
                if should_skip:
                    logger.info(
                        "Skipping optimizer update for zero-variance group %s.",
                        group.question_id,
                    )
                else:
                    for policy_epoch in range(args.num_policy_epochs):
                        update_started = time.perf_counter()
                        update_metrics = train_on_rollout_group(
                            model=model,
                            group=group,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            collator=collator,
                            loss_function=loss_function,
                            device=policy_device,
                            max_policy_steps_per_microbatch=(
                                args.max_policy_steps_per_microbatch
                            ),
                            max_grad_norm=args.max_grad_norm,
                        )
                        global_step += 1
                        update_seconds = time.perf_counter() - update_started

                        update_record = {
                            "event": "optimizer_step",
                            "epoch": epoch_index,
                            "position_in_epoch": position_in_epoch,
                            "processed_groups": processed_groups,
                            "global_step": global_step,
                            "policy_epoch": policy_epoch,
                            "question_id": group.question_id,
                            "zero_variance": bool(group.zero_variance),
                            "update_seconds": update_seconds,
                            **update_metrics,
                            **cuda_memory_metrics(policy_device),
                        }
                        append_jsonl(metrics_path, update_record)
                        logger.info(
                            "Update step=%d qid=%s policy_epoch=%d "
                            "loss=%.6f policy=%.6f kl=%.6f ratio=%.4f "
                            "clip=%.4f grad=%.4f",
                            global_step,
                            group.question_id,
                            policy_epoch,
                            update_metrics["loss"],
                            update_metrics["policy_loss"],
                            update_metrics["kl_loss"],
                            update_metrics["mean_ratio"],
                            update_metrics["clip_fraction"],
                            update_metrics["grad_norm"],
                        )

            processed_groups += 1
            next_epoch = epoch_index
            next_position = position_in_epoch + 1
            if next_position >= len(order):
                next_epoch = epoch_index + 1
                next_position = 0

            cursor_state = {
                "epoch_index": next_epoch,
                "next_position_in_epoch": next_position,
                "processed_groups": processed_groups,
                "global_step": global_step,
                "zero_variance_groups": zero_variance_groups,
                "elapsed_seconds": time.perf_counter() - training_started,
            }

            if (
                not args.rollout_only
                and not should_skip
                and global_step > 0
                and global_step % args.save_steps == 0
            ):
                checkpoint_dir = output_dir / f"checkpoint-{global_step:08d}"
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=cursor_state,
                    args=args,
                )
                rotate_checkpoints(output_dir, args.save_total_limit)

        position_start = 0
        if stop_training:
            break

    final_state = {
        **cursor_state,
        "processed_groups": processed_groups,
        "global_step": global_step,
        "zero_variance_groups": zero_variance_groups,
        "zero_variance_rate": (
            zero_variance_groups / float(max(processed_groups, 1))
        ),
        "elapsed_seconds": time.perf_counter() - training_started,
        "rollout_only": bool(args.rollout_only),
    }

    if not args.rollout_only:
        final_dir = output_dir / "final"
        save_checkpoint(
            checkpoint_dir=final_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            state=final_state,
            args=args,
        )
    else:
        write_json(output_dir / "rollout_only_state.json", final_state)

    append_jsonl(
        metrics_path,
        {
            "event": "training_complete",
            **final_state,
            **cuda_memory_metrics(policy_device),
        },
    )
    logger.info(
        "Finished. processed_groups=%d global_step=%d zero_variance_rate=%.4f output=%s",
        processed_groups,
        global_step,
        final_state["zero_variance_rate"],
        output_dir,
    )


if __name__ == "__main__":
    main()
