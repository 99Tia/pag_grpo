from __future__ import annotations

"""End-to-end rollout and GRPO-loss integration test.

This script runs one real MuSiQue question through the complete training path:

    trainable Llama-3 8B + LoRA controller
        -> G independent multi-step trajectories
        -> frozen SearchGraph/PPR + 70B services
        -> trajectory rewards and group-relative advantages
        -> policy-step collation
        -> clipped token-level GRPO loss
        -> one backward pass on controller JSON tokens only

By default, the test does *not* call optimizer.step(), so the LoRA weights are
not changed. Use ``--optimizer_step`` only when you deliberately want to verify
that one tiny update can be applied.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch import nn
from torch.optim import AdamW


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (str(SRC_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


# Reuse the exact construction functions used by the full trainer. This keeps
# the test and training paths synchronized rather than maintaining a second,
# slightly different environment/model setup.
import train_controller_grpo as train_lib  # noqa: E402
from ppr_agent.grpo_loss import (  # noqa: E402
    GRPOLoss,
    GRPOLossCollator,
    GRPOLossConfig,
)
from ppr_agent.grpo_rollout import collect_policy_step_samples  # noqa: E402
from ppr_agent.grpo_types import PolicyStepSample, RolloutGroup  # noqa: E402


logger = logging.getLogger("test_grpo_rollout")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = train_lib.build_arg_parser()
    parser.description = (
        "Run one end-to-end GRPO rollout/loss/backward integration test."
    )

    # Safe test defaults. Explicit CLI values still override these defaults.
    parser.set_defaults(
        limit=1,
        max_train_groups=1,
        num_train_epochs=1,
        shuffle=False,
        save_rollouts=True,
        rollout_only=False,
        group_size=2,
        num_policy_epochs=1,
        max_policy_steps_per_microbatch=1,
    )

    test_group = parser.add_argument_group("integration-test controls")
    test_group.add_argument(
        "--backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run loss.backward() after collecting the rollout group.",
    )
    test_group.add_argument(
        "--optimizer_step",
        action="store_true",
        help=(
            "Apply one AdamW step after backward. Disabled by default so this "
            "test does not modify the policy."
        ),
    )
    test_group.add_argument(
        "--require_nonzero_gradient",
        action="store_true",
        help=(
            "Fail when every trainable gradient is exactly zero. Do not use "
            "this for a zero-variance rollout group, where zero policy gradient "
            "can be mathematically correct."
        ),
    )
    test_group.add_argument(
        "--mean_ratio_warning_tolerance",
        type=float,
        default=0.05,
        help=(
            "Warn when the pre-update mean policy ratio differs from 1 by more "
            "than this amount."
        ),
    )
    test_group.add_argument(
        "--save_test_artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save test_rollout.json and test_summary.json under output_dir.",
    )
    test_group.add_argument(
        "--empty_cuda_cache_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Release unused cached CUDA blocks before exiting.",
    )
    return parser


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0**2)


def cuda_memory_snapshot() -> Dict[str, Any]:
    """Return allocations made by this Python process on each visible GPU."""

    if not torch.cuda.is_available():
        return {"cuda_available": False, "devices": []}

    devices: List[Dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "visible_index": index,
                "name": props.name,
                "total_mib": _mib(props.total_memory),
                "allocated_mib": _mib(torch.cuda.memory_allocated(index)),
                "reserved_mib": _mib(torch.cuda.memory_reserved(index)),
                "peak_allocated_mib": _mib(
                    torch.cuda.max_memory_allocated(index)
                ),
                "peak_reserved_mib": _mib(
                    torch.cuda.max_memory_reserved(index)
                ),
            }
        )
    return {"cuda_available": True, "devices": devices}


def reset_cuda_peaks() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        torch.cuda.reset_peak_memory_stats(index)


def gradient_diagnostics(model: nn.Module) -> Dict[str, Any]:
    trainable_tensors = 0
    tensors_with_grad = 0
    nonzero_grad_tensors = 0
    trainable_parameters = 0
    parameters_with_grad = 0
    squared_norm = 0.0
    max_abs_grad = 0.0

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        trainable_tensors += 1
        trainable_parameters += int(parameter.numel())
        if parameter.grad is None:
            continue

        tensors_with_grad += 1
        parameters_with_grad += int(parameter.numel())
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise FloatingPointError("A trainable parameter has a non-finite gradient.")

        grad_float = grad.float()
        tensor_norm = float(torch.linalg.vector_norm(grad_float).cpu())
        squared_norm += tensor_norm**2
        tensor_max = float(grad_float.abs().max().cpu()) if grad.numel() else 0.0
        max_abs_grad = max(max_abs_grad, tensor_max)
        if tensor_max > 0.0:
            nonzero_grad_tensors += 1

    return {
        "trainable_tensors": trainable_tensors,
        "tensors_with_grad": tensors_with_grad,
        "nonzero_grad_tensors": nonzero_grad_tensors,
        "trainable_parameters": trainable_parameters,
        "parameters_with_grad": parameters_with_grad,
        "global_grad_norm": math.sqrt(squared_norm),
        "max_abs_grad": max_abs_grad,
    }


def rollout_summary(group: RolloutGroup) -> Dict[str, Any]:
    trajectories: List[Dict[str, Any]] = []
    memory_ids: List[Any] = []

    for trajectory in group.trajectories:
        memory_id = trajectory.metadata.get("memory_identity")
        memory_ids.append(memory_id)
        trajectories.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "group_index": trajectory.group_index,
                "num_steps": trajectory.num_steps,
                "num_search_calls": trajectory.num_search_calls,
                "num_policy_tokens": trajectory.num_policy_tokens,
                "termination_reason": trajectory.termination_reason,
                "forced_stop": trajectory.forced_stop,
                "controller_final_answer": trajectory.controller_final_answer,
                "reader_predicted_answer": trajectory.reader_predicted_answer,
                "predicted_answer": trajectory.predicted_answer,
                "reward_total": float(trajectory.reward.total),
                "reward_components": dict(trajectory.reward.components),
                "advantage": float(trajectory.advantage or 0.0),
                "num_base_evidence": len(trajectory.base_evidence_passages),
                "num_fused_evidence": len(trajectory.fused_evidence_passages),
                "actions": [
                    {
                        "step_id": action.step_id,
                        "action_name": action.action_name,
                        "parse_success": action.parse_success,
                        "fallback_used": action.fallback_used,
                        "forced_stop": action.forced_stop,
                        "num_prompt_tokens": action.num_prompt_tokens,
                        "num_action_tokens": action.num_action_tokens,
                        "generated_text": action.generated_text,
                    }
                    for action in trajectory.actions
                ],
            }
        )

    non_null_memory_ids = [item for item in memory_ids if item is not None]
    independent_memories = (
        len(non_null_memory_ids) == len(set(non_null_memory_ids))
        if non_null_memory_ids
        else None
    )

    return {
        "question_id": group.question_id,
        "question": group.question,
        "group_size": group.group_size,
        "reward_mean": float(group.reward_mean or 0.0),
        "reward_std": float(group.reward_std or 0.0),
        "zero_variance": bool(group.zero_variance),
        "memory_identities": memory_ids,
        "independent_memory_objects": independent_memories,
        "trajectories": trajectories,
    }


# ---------------------------------------------------------------------------
# Backward-only microbatch path
# ---------------------------------------------------------------------------


def _chunks(
    values: Sequence[PolicyStepSample],
    size: int,
) -> Iterable[List[PolicyStepSample]]:
    if size <= 0:
        raise ValueError("Microbatch size must be greater than zero.")
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def backward_one_group(
    *,
    model: nn.Module,
    group: RolloutGroup,
    collator: GRPOLossCollator,
    loss_function: GRPOLoss,
    device: torch.device,
    max_policy_steps_per_microbatch: int,
) -> Dict[str, Any]:
    """Backpropagate one group without applying an optimizer step.

    The scaling exactly mirrors ``train_controller_grpo.py``:

        active tokens in microbatch
        ---------------------------  x  1 / number of trajectories
        active tokens in trajectory

    Thus, processing steps separately still recovers trajectory-mean weighting.
    """

    samples = collect_policy_step_samples([group])
    if not samples:
        raise ValueError("The rollout group produced no policy-step samples.")

    grouped = train_lib.group_samples_by_trajectory(samples)
    if not grouped:
        raise ValueError("No trajectories were available for backward testing.")

    model.zero_grad(set_to_none=True)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    metric_names = (
        "loss",
        "policy_loss",
        "kl_loss",
        "mean_reference_kl",
        "mean_ratio",
        "mean_log_ratio",
        "clip_fraction",
        "approximate_old_kl",
    )
    weighted_metrics = {name: 0.0 for name in metric_names}
    num_trajectories = len(grouped)
    num_microbatches = 0
    num_action_tokens = 0

    for trajectory_id, trajectory_samples in grouped.items():
        trajectory_tokens = sum(
            int(sum(sample.target_token_mask)) for sample in trajectory_samples
        )
        if trajectory_tokens <= 0:
            raise ValueError(
                f"Trajectory {trajectory_id!r} contains no active action tokens."
            )
        num_action_tokens += trajectory_tokens

        for micro_samples in _chunks(
            trajectory_samples,
            max_policy_steps_per_microbatch,
        ):
            batch = collator(micro_samples).to(device)
            micro_tokens = batch.num_action_tokens
            scale = (
                float(micro_tokens)
                / float(trajectory_tokens)
                / float(num_trajectories)
            )

            output = loss_function(
                policy_model=model,
                batch=batch,
                reference_model=None,
                reference_log_probs=None,
                return_log_probs=False,
            )
            if not torch.isfinite(output.loss):
                raise FloatingPointError("The GRPO loss is non-finite.")

            scaled_loss = output.loss * scale
            scaled_loss.backward()

            detached = output.detached_metrics()
            for name in metric_names:
                weighted_metrics[name] += float(detached[name]) * scale

            num_microbatches += 1
            del scaled_loss, output, batch

    gradients = gradient_diagnostics(model)
    weighted_metrics.update(
        {
            "num_trajectories": num_trajectories,
            "num_policy_steps": len(samples),
            "num_action_tokens": num_action_tokens,
            "num_microbatches": num_microbatches,
            "gradients": gradients,
        }
    )
    return weighted_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    train_lib.validate_args(args)

    if args.optimizer_step and not args.backward:
        parser.error("--optimizer_step requires --backward.")
    if args.mean_ratio_warning_tolerance < 0.0:
        parser.error("--mean_ratio_warning_tolerance must be non-negative.")

    train_lib.configure_logging(args.log_level)
    train_lib.set_global_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running GRPO integration test in Python: %s", sys.executable)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logger.info("Loading exactly one question from %s", args.questions_path)

    examples = train_lib.load_rollout_examples(
        args.questions_path,
        start=args.start,
        limit=1,
    )
    example = examples[0]

    resume_checkpoint = train_lib.resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        output_dir,
    )

    overall_start = time.perf_counter()
    stage_memory: Dict[str, Any] = {
        "before_model_load": cuda_memory_snapshot(),
    }

    policy, tokenizer = train_lib.build_trainable_policy(
        args,
        resume_checkpoint=resume_checkpoint,
    )
    model = policy.model
    policy_device = policy.input_device
    stage_memory["after_model_load"] = cuda_memory_snapshot()

    environment = train_lib.build_frozen_environment(
        args,
        prompt_agent=policy.reasoning_agent,
    )
    reward_calculator = train_lib.build_reward_calculator(args)
    collector = train_lib.build_rollout_collector(
        args,
        policy=policy,
        environment=environment,
        reward_calculator=reward_calculator,
    )
    stage_memory["after_environment_load"] = cuda_memory_snapshot()

    reset_cuda_peaks()
    rollout_start = time.perf_counter()
    group = collector.collect_group(example, rollout_iteration=0)
    rollout_seconds = time.perf_counter() - rollout_start
    group.validate(require_old_log_probs=True)
    stage_memory["after_rollout"] = cuda_memory_snapshot()

    samples = collect_policy_step_samples([group])
    if not samples:
        raise RuntimeError("No policy-step samples were created from the rollout.")
    for sample in samples:
        sample.validate()

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

    backward_metrics: Dict[str, Any] = {"backward_executed": False}
    optimizer_changed_parameter = None

    if args.backward:
        reset_cuda_peaks()
        backward_start = time.perf_counter()
        backward_metrics = backward_one_group(
            model=model,
            group=group,
            collator=collator,
            loss_function=loss_function,
            device=policy_device,
            max_policy_steps_per_microbatch=(
                args.max_policy_steps_per_microbatch
            ),
        )
        backward_metrics["backward_executed"] = True
        backward_metrics["backward_seconds"] = (
            time.perf_counter() - backward_start
        )
        stage_memory["after_backward"] = cuda_memory_snapshot()

        gradients = backward_metrics["gradients"]
        if int(gradients["tensors_with_grad"]) <= 0:
            raise RuntimeError(
                "Backward completed, but no trainable LoRA parameter received a gradient."
            )
        if args.require_nonzero_gradient and int(
            gradients["nonzero_grad_tensors"]
        ) <= 0:
            raise RuntimeError(
                "All trainable gradients are zero. This can be expected for a "
                "zero-variance group with zero reference KL, otherwise inspect "
                "the action mask and advantage values."
            )

        ratio_error = abs(float(backward_metrics["mean_ratio"]) - 1.0)
        backward_metrics["mean_ratio_error_from_one"] = ratio_error
        if ratio_error > args.mean_ratio_warning_tolerance:
            logger.warning(
                "Pre-update mean policy ratio %.6f differs from 1 by %.6f. "
                "This may indicate rollout/current token alignment drift.",
                float(backward_metrics["mean_ratio"]),
                ratio_error,
            )

        if args.optimizer_step:
            first_parameter = next(
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            before = first_parameter.detach().float().cpu().clone()
            optimizer = AdamW(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay,
            )
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ],
                max_norm=args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            after = first_parameter.detach().float().cpu()
            optimizer_changed_parameter = bool(not torch.equal(before, after))
            backward_metrics["optimizer_step_executed"] = True
            backward_metrics["optimizer_changed_first_trainable_parameter"] = (
                optimizer_changed_parameter
            )
        else:
            backward_metrics["optimizer_step_executed"] = False
            # Avoid retaining gradient buffers after the test.
            model.zero_grad(set_to_none=True)

    summary = {
        "status": "passed",
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy_device": str(policy_device),
        "embedding_device": args.embedding_device,
        "model_name_or_path": args.model_name_or_path,
        "question_id": example.question_id,
        "group_size": group.group_size,
        "rollout_seconds": rollout_seconds,
        "total_seconds": time.perf_counter() - overall_start,
        "num_policy_step_samples": len(samples),
        "rollout": rollout_summary(group),
        "backward": backward_metrics,
        "cuda_memory": stage_memory,
        "optimizer_changed_parameter": optimizer_changed_parameter,
        "notes": {
            "zero_variance_group": bool(group.zero_variance),
            "zero_variance_explanation": (
                "When every trajectory has the same reward, all group-relative "
                "advantages are zero. A zero policy gradient can then be correct."
            ),
            "weights_modified": bool(args.optimizer_step),
        },
    }

    if args.save_test_artifacts:
        group.save_json(output_dir / "test_rollout.json")
        write_json(output_dir / "test_summary.json", summary)

    logger.info("GRPO integration test PASSED")
    logger.info(
        "qid=%s G=%d rewards=%s advantages=%s zero_variance=%s",
        group.question_id,
        group.group_size,
        [round(float(t.reward.total), 6) for t in group.trajectories],
        [round(float(t.advantage or 0.0), 6) for t in group.trajectories],
        bool(group.zero_variance),
    )
    logger.info(
        "policy_steps=%d action_tokens=%d rollout_seconds=%.2f",
        len(samples),
        sum(int(sum(sample.target_token_mask)) for sample in samples),
        rollout_seconds,
    )

    if args.backward:
        logger.info(
            "loss=%.6f policy_loss=%.6f kl_loss=%.6f mean_ratio=%.6f "
            "clip_fraction=%.6f grad_norm=%.6f",
            float(backward_metrics["loss"]),
            float(backward_metrics["policy_loss"]),
            float(backward_metrics["kl_loss"]),
            float(backward_metrics["mean_ratio"]),
            float(backward_metrics["clip_fraction"]),
            float(backward_metrics["gradients"]["global_grad_norm"]),
        )

    if args.save_test_artifacts:
        logger.info("Saved %s", output_dir / "test_rollout.json")
        logger.info("Saved %s", output_dir / "test_summary.json")

    if args.empty_cuda_cache_at_end and torch.cuda.is_available():
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError as exc:
        logger.exception("CUDA out of memory during the GRPO integration test.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            "\nSuggested smaller debug settings:\n"
            "  --group_size 2\n"
            "  --max_policy_steps_per_microbatch 1\n"
            "  --max_prompt_tokens 2048\n"
            "  --max_action_tokens 256\n"
            "  --embedding_device cpu\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
