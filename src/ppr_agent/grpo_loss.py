from __future__ import annotations
import contextlib
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
from .grpo_types import PolicyStepSample

ReductionMode = Literal["trajectory_mean", "sample_mean", "token_mean"]


@dataclass(frozen=True)
class GRPOLossConfig:
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    beta: float = 0.01
    reduction: ReductionMode = "trajectory_mean"
    max_abs_log_ratio: float = 20.0
    use_reference_kl: bool = True
    apply_kl_when_advantage_is_zero: bool = True
    validate_trajectory_consistency: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon_low < 1.0:
            raise ValueError("epsilon_low must satisfy 0 <= epsilon_low < 1.")
        if self.epsilon_high < 0.0:
            raise ValueError("epsilon_high must be non-negative.")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if self.reduction not in ("trajectory_mean", "sample_mean", "token_mean"):
            raise ValueError(f"Unsupported reduction: {self.reduction!r}.")
        if self.max_abs_log_ratio <= 0.0:
            raise ValueError("max_abs_log_ratio must be greater than zero.")


@dataclass
class GRPOLossBatch:

    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor

    target_token_ids: Tensor
    action_mask: Tensor
    old_log_probs: Tensor

    advantages: Tensor
    trajectory_rewards: Tensor

    trajectory_ids: List[str]
    question_ids: List[str]
    group_indices: List[int]
    step_ids: List[int]

    prompt_lengths: Tensor
    target_lengths: Tensor
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def max_sequence_length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def max_target_length(self) -> int:
        return int(self.target_token_ids.shape[1])

    @property
    def num_action_tokens(self) -> int:
        return int(self.action_mask.sum().item())

    def to(self, device: torch.device | str) -> "GRPOLossBatch":
        return GRPOLossBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device),
            target_token_ids=self.target_token_ids.to(device),
            action_mask=self.action_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            advantages=self.advantages.to(device),
            trajectory_rewards=self.trajectory_rewards.to(device),
            trajectory_ids=list(self.trajectory_ids),
            question_ids=list(self.question_ids),
            group_indices=list(self.group_indices),
            step_ids=list(self.step_ids),
            prompt_lengths=self.prompt_lengths.to(device),
            target_lengths=self.target_lengths.to(device),
            metadata=[dict(item) for item in self.metadata],
        )

    def validate(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, S).")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        if self.position_ids.shape != self.input_ids.shape:
            raise ValueError("position_ids must match input_ids shape.")

        batch_size = self.input_ids.shape[0]
        target_shape = self.target_token_ids.shape
        if len(target_shape) != 2 or target_shape[0] != batch_size:
            raise ValueError("target_token_ids must have shape (B, T).")
        if self.action_mask.shape != target_shape:
            raise ValueError("action_mask must match target_token_ids shape.")
        if self.old_log_probs.shape != target_shape:
            raise ValueError("old_log_probs must match target_token_ids shape.")

        for name, tensor in (
            ("advantages", self.advantages),
            ("trajectory_rewards", self.trajectory_rewards),
            ("prompt_lengths", self.prompt_lengths),
            ("target_lengths", self.target_lengths),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape (B,).")

        for name, values in (
            ("trajectory_ids", self.trajectory_ids),
            ("question_ids", self.question_ids),
            ("group_indices", self.group_indices),
            ("step_ids", self.step_ids),
            ("metadata", self.metadata),
        ):
            if len(values) != batch_size:
                raise ValueError(f"{name} length must equal batch size.")

        if self.max_target_length <= 0:
            raise ValueError("At least one target token is required.")
        if self.num_action_tokens <= 0:
            raise ValueError("The batch contains no unmasked controller tokens.")

        if not torch.all((self.attention_mask == 0) | (self.attention_mask == 1)):
            raise ValueError("attention_mask must be binary.")
        if not torch.all((self.action_mask == 0) | (self.action_mask == 1)):
            raise ValueError("action_mask must be binary.")

        active_old = self.old_log_probs[self.action_mask.bool()]
        if not torch.isfinite(active_old).all():
            raise ValueError("old_log_probs contain non-finite active values.")
        if not torch.isfinite(self.advantages).all():
            raise ValueError("advantages contain non-finite values.")
        if not torch.isfinite(self.trajectory_rewards).all():
            raise ValueError("trajectory_rewards contain non-finite values.")


@dataclass
class GRPOLossOutput:

    loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor

    mean_reference_kl: Tensor
    mean_ratio: Tensor
    mean_log_ratio: Tensor
    clip_fraction: Tensor
    approximate_old_kl: Tensor

    num_action_tokens: int
    num_policy_steps: int
    num_trajectories: int

    current_log_probs: Optional[Tensor] = None
    reference_log_probs: Optional[Tensor] = None

    def detached_metrics(self) -> Dict[str, float]:
        return {
            "loss": float(self.loss.detach().cpu()),
            "policy_loss": float(self.policy_loss.detach().cpu()),
            "kl_loss": float(self.kl_loss.detach().cpu()),
            "mean_reference_kl": float(self.mean_reference_kl.detach().cpu()),
            "mean_ratio": float(self.mean_ratio.detach().cpu()),
            "mean_log_ratio": float(self.mean_log_ratio.detach().cpu()),
            "clip_fraction": float(self.clip_fraction.detach().cpu()),
            "approximate_old_kl": float(self.approximate_old_kl.detach().cpu()),
            "num_action_tokens": float(self.num_action_tokens),
            "num_policy_steps": float(self.num_policy_steps),
            "num_trajectories": float(self.num_trajectories),
        }

class GRPOLossCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        pad_to_multiple_of: Optional[int] = 8,
        validate_samples: bool = True,
    ) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative.")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive or None.")

        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.validate_samples = validate_samples

    def __call__(self, samples: Sequence[PolicyStepSample]) -> GRPOLossBatch:
        if not samples:
            raise ValueError("Cannot collate an empty list of policy-step samples.")

        if self.validate_samples:
            for index, sample in enumerate(samples):
                try:
                    sample.validate()
                except Exception as exc:
                    raise ValueError(f"Invalid PolicyStepSample at index {index}: {exc}") from exc

        sequence_lengths = [
            len(sample.prompt_input_ids) + len(sample.target_token_ids)
            for sample in samples
        ]
        prompt_lengths = [len(sample.prompt_input_ids) for sample in samples]
        target_lengths = [len(sample.target_token_ids) for sample in samples]

        if any(length <= 0 for length in prompt_lengths):
            raise ValueError("Every sample must contain at least one prompt token.")
        if any(length <= 0 for length in target_lengths):
            raise ValueError("Every sample must contain at least one generated token.")

        max_sequence_length = _round_up(
            max(sequence_lengths), self.pad_to_multiple_of
        )
        max_target_length = max(target_lengths)
        batch_size = len(samples)

        input_ids = torch.full(
            (batch_size, max_sequence_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_sequence_length), dtype=torch.long
        )

        target_token_ids = torch.full(
            (batch_size, max_target_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        action_mask = torch.zeros(
            (batch_size, max_target_length), dtype=torch.float32
        )
        old_log_probs = torch.zeros(
            (batch_size, max_target_length), dtype=torch.float32
        )

        advantages = torch.empty(batch_size, dtype=torch.float32)
        trajectory_rewards = torch.empty(batch_size, dtype=torch.float32)

        trajectory_ids: List[str] = []
        question_ids: List[str] = []
        group_indices: List[int] = []
        step_ids: List[int] = []
        metadata: List[Dict[str, Any]] = []

        for row, sample in enumerate(samples):
            full_ids = list(sample.prompt_input_ids) + list(sample.target_token_ids)
            sequence_length = len(full_ids)
            sequence_start = max_sequence_length - sequence_length

            input_ids[row, sequence_start:] = torch.tensor(full_ids, dtype=torch.long)
            attention_mask[row, sequence_start:] = 1

            target_length = len(sample.target_token_ids)
            target_start = max_target_length - target_length
            target_token_ids[row, target_start:] = torch.tensor(
                sample.target_token_ids, dtype=torch.long
            )
            action_mask[row, target_start:] = torch.tensor(
                sample.target_token_mask, dtype=torch.float32
            )
            old_log_probs[row, target_start:] = torch.tensor(
                sample.old_log_probs, dtype=torch.float32
            )

            advantages[row] = float(sample.advantage)
            trajectory_rewards[row] = float(sample.trajectory_reward)

            trajectory_ids.append(str(sample.trajectory_id))
            question_ids.append(str(sample.question_id))
            group_indices.append(int(sample.group_index))
            step_ids.append(int(sample.step_id))
            metadata.append(dict(sample.metadata))

        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        batch = GRPOLossBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            target_token_ids=target_token_ids,
            action_mask=action_mask,
            old_log_probs=old_log_probs,
            advantages=advantages,
            trajectory_rewards=trajectory_rewards,
            trajectory_ids=trajectory_ids,
            question_ids=question_ids,
            group_indices=group_indices,
            step_ids=step_ids,
            prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long),
            target_lengths=torch.tensor(target_lengths, dtype=torch.long),
            metadata=metadata,
        )
        batch.validate()
        return batch


def compute_action_log_probs(model: nn.Module, batch: GRPOLossBatch) -> Tensor:
    batch.validate()
    logits_to_keep = batch.max_target_length + 1

    outputs = _forward_with_optional_logits_window(
        model=model,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        logits_to_keep=logits_to_keep,
    )

    logits = outputs.logits
    if logits.ndim != 3:
        raise RuntimeError(
            f"Model logits must have shape (B, S, V), got {tuple(logits.shape)}."
        )
    if logits.shape[0] != batch.batch_size:
        raise RuntimeError("Model logits batch dimension does not match the loss batch.")
    if logits.shape[1] < logits_to_keep:
        raise RuntimeError(
            "Model returned too few logits for action alignment: "
            f"needed {logits_to_keep}, got {int(logits.shape[1])}."
        )

    logits = logits[:, -logits_to_keep:, :]
    action_prediction_logits = logits[:, :-1, :]

    if action_prediction_logits.shape[1] != batch.max_target_length:
        raise RuntimeError("Action-logit window alignment failed.")

    return _selective_log_softmax(
        action_prediction_logits,
        batch.target_token_ids,
    )


def _forward_with_optional_logits_window(
    *,
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    logits_to_keep: int,
) -> Any:
    base_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": False,
        "return_dict": True,
    }

    candidate_names = _candidate_logits_window_kwargs(model)
    last_unexpected_keyword_error: Optional[TypeError] = None

    for keyword in candidate_names:
        try:
            return model(**base_kwargs, **{keyword: logits_to_keep})
        except TypeError as exc:
            if _is_unexpected_keyword_error(exc, keyword):
                last_unexpected_keyword_error = exc
                continue
            raise


    try:
        return model(**base_kwargs)
    except TypeError:
   
        minimal_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        try:
            return model(**minimal_kwargs)
        except Exception:
            if last_unexpected_keyword_error is not None:
                raise last_unexpected_keyword_error
            raise


def _candidate_logits_window_kwargs(model: nn.Module) -> List[str]:
    names: List[str] = []
    try:
        signature = inspect.signature(model.forward)
        parameters = signature.parameters
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in parameters:
                names.append(name)
    except (TypeError, ValueError):
        pass
    names.extend(["logits_to_keep", "num_logits_to_keep"])
    return list(dict.fromkeys(names))


def _is_unexpected_keyword_error(exc: TypeError, keyword: str) -> bool:
    text = str(exc)
    return keyword in text and (
        "unexpected keyword" in text
        or "unexpected keyword argument" in text
        or "got an unexpected" in text
    )


def _selective_log_softmax(logits: Tensor, token_ids: Tensor) -> Tensor:
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            "logits leading dimensions must match token_ids: "
            f"{tuple(logits.shape[:-1])} != {tuple(token_ids.shape)}."
        )

    logits_f32 = logits.float()
    selected_logits = torch.gather(
        logits_f32,
        dim=-1,
        index=token_ids.unsqueeze(-1),
    ).squeeze(-1)
    log_normalizer = torch.logsumexp(logits_f32, dim=-1)
    return selected_logits - log_normalizer

@contextlib.contextmanager
def reference_policy_context(
    policy_model: nn.Module,
    reference_model: Optional[nn.Module] = None,
):
    if reference_model is not None:
        previous_training = reference_model.training
        reference_model.eval()
        try:
            yield reference_model
        finally:
            reference_model.train(previous_training)
        return

    disable_adapter = getattr(policy_model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise ValueError(
            "Reference KL is enabled, but no reference_model was supplied and "
            "the policy model does not expose disable_adapter(). Load the policy "
            "as a PEFT/LoRA model or pass a frozen reference model explicitly."
        )

    previous_training = policy_model.training
    policy_model.eval()
    try:
        with disable_adapter():
            yield policy_model
    finally:
        policy_model.train(previous_training)

class GRPOLoss(nn.Module):
    def __init__(self, config: Optional[GRPOLossConfig] = None) -> None:
        super().__init__()
        self.config = config or GRPOLossConfig()

    def forward(
        self,
        policy_model: nn.Module,
        batch: GRPOLossBatch,
        *,
        reference_model: Optional[nn.Module] = None,
        reference_log_probs: Optional[Tensor] = None,
        return_log_probs: bool = False,
    ) -> GRPOLossOutput:
        batch.validate()
        if self.config.validate_trajectory_consistency:
            _validate_trajectory_values(batch)

        current_log_probs = compute_action_log_probs(policy_model, batch)
        if not torch.isfinite(current_log_probs[batch.action_mask.bool()]).all():
            raise FloatingPointError("Current policy produced non-finite log-probabilities.")

        old_log_probs = batch.old_log_probs.to(current_log_probs.dtype)
        action_mask = batch.action_mask.to(current_log_probs.dtype)
        advantages = batch.advantages.to(current_log_probs.dtype).unsqueeze(1)

        raw_log_ratio = current_log_probs - old_log_probs
        stable_log_ratio = raw_log_ratio.clamp(
            min=-self.config.max_abs_log_ratio,
            max=self.config.max_abs_log_ratio,
        )
        ratios = stable_log_ratio.exp()

        unclipped_objective = ratios * advantages
        clipped_ratios = ratios.clamp(
            min=1.0 - self.config.epsilon_low,
            max=1.0 + self.config.epsilon_high,
        )
        clipped_objective = clipped_ratios * advantages
        per_token_policy_loss = -torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        need_reference = (
            self.config.use_reference_kl and self.config.beta > 0.0
        )
        ref_log_probs: Optional[Tensor]
        if need_reference:
            if reference_log_probs is None:
                with torch.no_grad(), reference_policy_context(
                    policy_model=policy_model,
                    reference_model=reference_model,
                ) as ref_policy:
                    ref_log_probs = compute_action_log_probs(ref_policy, batch)
            else:
                ref_log_probs = reference_log_probs.to(
                    device=current_log_probs.device,
                    dtype=current_log_probs.dtype,
                )
                if ref_log_probs.shape != current_log_probs.shape:
                    raise ValueError(
                        "reference_log_probs must match current log-probability shape: "
                        f"{tuple(ref_log_probs.shape)} != {tuple(current_log_probs.shape)}."
                    )

            if not torch.isfinite(ref_log_probs[action_mask.bool()]).all():
                raise FloatingPointError(
                    "Reference policy produced non-finite log-probabilities."
                )

            ref_minus_current = ref_log_probs - current_log_probs
            per_token_reference_kl = (
                torch.exp(
                    ref_minus_current.clamp(
                        min=-self.config.max_abs_log_ratio,
                        max=self.config.max_abs_log_ratio,
                    )
                )
                - ref_minus_current
                - 1.0
            )

            kl_mask = action_mask
            if not self.config.apply_kl_when_advantage_is_zero:
                nonzero_advantage = (advantages.abs() > 0.0).to(action_mask.dtype)
                kl_mask = kl_mask * nonzero_advantage
        else:
            ref_log_probs = None
            per_token_reference_kl = torch.zeros_like(per_token_policy_loss)
            kl_mask = action_mask

        policy_loss = _reduce_masked_values(
            values=per_token_policy_loss,
            mask=action_mask,
            batch=batch,
            reduction=self.config.reduction,
        )

        if need_reference and float(kl_mask.sum().item()) > 0.0:
            raw_kl_loss = _reduce_masked_values(
                values=per_token_reference_kl,
                mask=kl_mask,
                batch=batch,
                reduction=self.config.reduction,
            )
        else:
            raw_kl_loss = current_log_probs.sum() * 0.0

        kl_loss = float(self.config.beta) * raw_kl_loss
        total_loss = policy_loss + kl_loss

        with torch.no_grad():
            active = action_mask.bool()
            mean_ratio = _masked_mean(ratios, action_mask)
            mean_log_ratio = _masked_mean(raw_log_ratio, action_mask)
            mean_reference_kl = _masked_mean(
                per_token_reference_kl,
                kl_mask,
                allow_empty=True,
            )

            clipped = (
                (ratios < (1.0 - self.config.epsilon_low))
                | (ratios > (1.0 + self.config.epsilon_high))
            ).to(action_mask.dtype)
            clip_fraction = _masked_mean(clipped, action_mask)

            # Common PPO diagnostic between old rollout policy and current
            # policy. It is not the reference-model KL regularizer.
            approximate_old_kl = _masked_mean(
                (ratios - 1.0) - raw_log_ratio,
                action_mask,
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError("GRPO loss became non-finite.")
            if active.sum() == 0:
                raise ValueError("No active controller tokens were available for loss.")

        return GRPOLossOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            mean_reference_kl=mean_reference_kl,
            mean_ratio=mean_ratio,
            mean_log_ratio=mean_log_ratio,
            clip_fraction=clip_fraction,
            approximate_old_kl=approximate_old_kl,
            num_action_tokens=batch.num_action_tokens,
            num_policy_steps=batch.batch_size,
            num_trajectories=len(set(batch.trajectory_ids)),
            current_log_probs=(current_log_probs if return_log_probs else None),
            reference_log_probs=(ref_log_probs if return_log_probs else None),
        )

def _reduce_masked_values(
    *,
    values: Tensor,
    mask: Tensor,
    batch: GRPOLossBatch,
    reduction: ReductionMode,
) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shape.")

    masked_values = values * mask

    if reduction == "token_mean":
        denominator = mask.sum().clamp_min(1.0)
        return masked_values.sum() / denominator

    row_denominator = mask.sum(dim=1).clamp_min(1.0)
    row_means = masked_values.sum(dim=1) / row_denominator
    active_rows = mask.sum(dim=1) > 0

    if reduction == "sample_mean":
        if not active_rows.any():
            raise ValueError("No active rows for sample_mean reduction.")
        return row_means[active_rows].mean()

    if reduction != "trajectory_mean":
        raise ValueError(f"Unsupported reduction: {reduction!r}.")

    trajectory_order = list(dict.fromkeys(batch.trajectory_ids))
    trajectory_losses: List[Tensor] = []
    for trajectory_id in trajectory_order:
        row_indices = [
            index
            for index, value in enumerate(batch.trajectory_ids)
            if value == trajectory_id
        ]
        index_tensor = torch.tensor(
            row_indices,
            dtype=torch.long,
            device=values.device,
        )
        trajectory_sum = masked_values.index_select(0, index_tensor).sum()
        trajectory_count = mask.index_select(0, index_tensor).sum()
        if float(trajectory_count.detach().item()) <= 0.0:
            continue
        trajectory_losses.append(trajectory_sum / trajectory_count)

    if not trajectory_losses:
        raise ValueError("No active trajectories for trajectory_mean reduction.")
    return torch.stack(trajectory_losses).mean()


def _masked_mean(values: Tensor, mask: Tensor, allow_empty: bool = False) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shape.")
    denominator = mask.sum()
    if float(denominator.detach().item()) <= 0.0:
        if allow_empty:
            return values.sum() * 0.0
        raise ValueError("Cannot compute a masked mean over an empty mask.")
    return (values * mask).sum() / denominator


def _validate_trajectory_values(batch: GRPOLossBatch) -> None:
    seen: Dict[str, Tuple[float, float, str, int]] = {}
    for index, trajectory_id in enumerate(batch.trajectory_ids):
        advantage = float(batch.advantages[index].detach().cpu())
        reward = float(batch.trajectory_rewards[index].detach().cpu())
        question_id = batch.question_ids[index]
        group_index = batch.group_indices[index]

        previous = seen.get(trajectory_id)
        if previous is None:
            seen[trajectory_id] = (advantage, reward, question_id, group_index)
            continue

        old_advantage, old_reward, old_question_id, old_group_index = previous
        if not math.isclose(advantage, old_advantage, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Trajectory {trajectory_id!r} has inconsistent advantages across steps."
            )
        if not math.isclose(reward, old_reward, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Trajectory {trajectory_id!r} has inconsistent rewards across steps."
            )
        if question_id != old_question_id or group_index != old_group_index:
            raise ValueError(
                f"Trajectory {trajectory_id!r} has inconsistent question/group metadata."
            )


def _round_up(value: int, multiple: Optional[int]) -> int:
    if multiple is None:
        return int(value)
    return int(math.ceil(value / multiple) * multiple)


__all__ = [
    "GRPOLoss",
    "GRPOLossBatch",
    "GRPOLossCollator",
    "GRPOLossConfig",
    "GRPOLossOutput",
    "compute_action_log_probs",
    "reference_policy_context",
]
