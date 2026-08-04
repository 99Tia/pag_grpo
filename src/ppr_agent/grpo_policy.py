from __future__ import annotations
import contextlib
import logging
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from .grpo_types import RolloutAction
from .reasoning_agent import (
    ReasoningAgent,
    normalize_action_name,
    parse_json_like_output,
)
from .schema import AgentAction

logger = logging.getLogger(__name__)

DeviceMap = Union[str, Dict[str, Union[int, str]]]


@dataclass(frozen=True)
class GRPOPolicyConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    model_revision: str = "main"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    device: str = "cuda"
    device_map: Optional[DeviceMap] = None
    low_cpu_mem_usage: bool = True
    # Exact controller-prefix handling.
    max_prompt_tokens: Optional[int] = 4096
    add_generation_prompt: bool = True
    # Rollout sampling. Temperature 0.7 provides diversity across a GRPO group.
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    do_sample: Optional[bool] = None
    use_cache: bool = True
    include_eos_in_action_mask: bool = True
    store_full_prompt_text_in_metadata: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty.")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive or None.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if self.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens.")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero.")

    @property
    def sampling_enabled(self) -> bool:
        if self.do_sample is not None:
            return bool(self.do_sample)
        return self.temperature > 0.0


@dataclass
class PromptEncoding:
    messages: List[Dict[str, str]]
    full_prompt_text: str
    effective_prompt_text: str
    input_ids: Tensor
    attention_mask: Tensor
    original_num_tokens: int
    truncated: bool

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.shape[-1])


@dataclass
class SampledPolicyAction:
    agent_action: AgentAction
    rollout_action: RolloutAction

    def validate(self) -> None:
        self.agent_action.validate()
        self.rollout_action.validate(require_old_log_probs=True)


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _to_plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).strip().lower().replace("torch.", "")
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
            f"Unsupported torch_dtype={dtype_name!r}. "
            f"Choose one of {sorted(aliases)}."
        )
    return aliases[normalized]


def _normalize_eos_ids(eos_token_id: Any) -> List[int]:
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, str):
        return [int(value) for value in eos_token_id]
    raise TypeError(f"Unsupported eos_token_id type: {type(eos_token_id)!r}.")


def _strict_action_format_diagnostics(raw_output: str) -> Dict[str, Any]:
    parsed = parse_json_like_output(raw_output)
    errors: List[str] = []

    if not parsed:
        return {
            "parsed_json": {},
            "requested_action_name": None,
            "recognized_action": False,
            "strict_format_valid": False,
            "fallback_required": True,
            "format_errors": ["No JSON object could be parsed."],
        }

    requested_name = normalize_action_name(parsed.get("action", ""))
    recognized = requested_name in ("SearchGraph", "SubmitFinalAnswer")

    if not recognized:
        errors.append("The 'action' field is missing or unsupported.")
    elif requested_name == "SearchGraph":
        search_focus = parsed.get("search_focus")
        seed_entities = parsed.get("seed_entities")
        relation_hints = parsed.get("relation_hints")

        if not isinstance(search_focus, str) or not search_focus.strip():
            errors.append("SearchGraph requires a non-empty string search_focus.")
        if not isinstance(seed_entities, list):
            errors.append("SearchGraph requires seed_entities to be a list.")
        if not isinstance(relation_hints, list):
            errors.append("SearchGraph requires relation_hints to be a list.")
    elif requested_name == "SubmitFinalAnswer":
        answer = parsed.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            errors.append("SubmitFinalAnswer requires a non-empty string answer.")

    fallback_required = not recognized
    if requested_name == "SubmitFinalAnswer":
        fallback_required = fallback_required or not str(parsed.get("answer", "")).strip()

    return {
        "parsed_json": _to_plain(parsed),
        "requested_action_name": requested_name if recognized else None,
        "recognized_action": bool(recognized),
        "strict_format_valid": bool(recognized and not errors),
        "fallback_required": bool(fallback_required),
        "format_errors": errors,
    }


class GRPOPolicy:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        reasoning_agent: ReasoningAgent,
        config: GRPOPolicyConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reasoning_agent = reasoning_agent
        self.config = config

        self._prepare_tokenizer_and_model()
        self.input_device = self._infer_input_device()

    @classmethod
    def from_pretrained(
        cls,
        *,
        config: GRPOPolicyConfig,
        reasoning_agent: ReasoningAgent,
        model_load_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "GRPOPolicy":
        tokenizer_kwargs = dict(tokenizer_load_kwargs or {})
        tokenizer_kwargs.setdefault("revision", config.model_revision)
        tokenizer_kwargs.setdefault("trust_remote_code", config.trust_remote_code)

        tokenizer_name = config.tokenizer_name_or_path or config.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

        model_kwargs = dict(model_load_kwargs or {})
        model_kwargs.setdefault("revision", config.model_revision)
        model_kwargs.setdefault("trust_remote_code", config.trust_remote_code)
        model_kwargs.setdefault("torch_dtype", _resolve_torch_dtype(config.torch_dtype))
        model_kwargs.setdefault("low_cpu_mem_usage", config.low_cpu_mem_usage)

        if config.attn_implementation:
            model_kwargs.setdefault("attn_implementation", config.attn_implementation)
        if config.device_map is not None:
            model_kwargs.setdefault("device_map", config.device_map)

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )

        if config.device_map is None:
            model = model.to(torch.device(config.device))

        return cls(
            model=model,
            tokenizer=tokenizer,
            reasoning_agent=reasoning_agent,
            config=config,
        )

    def _prepare_tokenizer_and_model(self) -> None:
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise ValueError(
                "The controller tokenizer has no chat_template. Set the model's "
                "Llama chat template before collecting GRPO rollouts."
            )

        if self.tokenizer.eos_token_id is None:
            raise ValueError("The controller tokenizer must define eos_token_id.")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.use_cache = self.config.use_cache

        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id
            generation_config.eos_token_id = self.tokenizer.eos_token_id

    def _infer_input_device(self) -> torch.device:
        model_device = getattr(self.model, "device", None)
        if isinstance(model_device, torch.device) and model_device.type != "meta":
            return model_device

        try:
            for parameter in self.model.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
        except Exception as exc:  # pragma: no cover - defensive for wrappers
            logger.debug("Could not infer device from model parameters: %s", exc)

        return torch.device(self.config.device)

    def _model_context_limit(self) -> Optional[int]:
        candidates: List[int] = []
        model_config = getattr(self.model, "config", None)
        for name in (
            "max_position_embeddings",
            "max_sequence_length",
            "n_positions",
            "seq_length",
        ):
            value = getattr(model_config, name, None)
            if isinstance(value, int) and 0 < value < 10_000_000:
                candidates.append(value)

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
            candidates.append(tokenizer_limit)

        return min(candidates) if candidates else None

    def _effective_prompt_limit(self) -> Optional[int]:
        configured = self.config.max_prompt_tokens
        context_limit = self._model_context_limit()

        if context_limit is None:
            return configured

        available = context_limit - self.config.max_new_tokens
        if available <= 0:
            raise ValueError(
                "max_new_tokens leaves no room for a prompt: "
                f"context_limit={context_limit}, "
                f"max_new_tokens={self.config.max_new_tokens}."
            )

        return available if configured is None else min(configured, available)

    def encode_messages(self, messages: Sequence[Mapping[str, str]]) -> PromptEncoding:
        normalized_messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]

        full_prompt_text = self.tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=self.config.add_generation_prompt,
        )

        encoded = self.tokenizer(
            full_prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        original_num_tokens = int(input_ids.shape[-1])

        prompt_limit = self._effective_prompt_limit()
        truncated = bool(prompt_limit is not None and original_num_tokens > prompt_limit)
        if truncated:
            input_ids = input_ids[:, -prompt_limit:]
            attention_mask = attention_mask[:, -prompt_limit:]

        input_ids = input_ids.to(self.input_device)
        attention_mask = attention_mask.to(self.input_device)

        effective_prompt_text = self.tokenizer.decode(
            input_ids[0].detach().cpu().tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        return PromptEncoding(
            messages=normalized_messages,
            full_prompt_text=full_prompt_text,
            effective_prompt_text=effective_prompt_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            original_num_tokens=original_num_tokens,
            truncated=truncated,
        )

    def build_prompt(self, *, question: str, step_id: int, evidence_memory: Optional[Any],) -> PromptEncoding:
        messages = self.reasoning_agent.build_messages(
            question=question,
            step_id=step_id,
            evidence_memory=evidence_memory,
        )
        return self.encode_messages(messages)

    @contextlib.contextmanager
    def _rollout_mode(self) -> Iterator[None]:
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    @contextlib.contextmanager
    def _seeded_rng(self, seed: Optional[int]) -> Iterator[None]:
        if seed is None:
            yield
            return

        cuda_devices: List[int] = []
        if self.input_device.type == "cuda":
            device_index = self.input_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices = [int(device_index)]

        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed(int(seed))
            yield

    def _generation_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "min_new_tokens": self.config.min_new_tokens,
            "do_sample": self.config.sampling_enabled,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.config.use_cache,
            "return_dict_in_generate": True,
        }

        if self.config.sampling_enabled:
            kwargs.update(
                {
                    "temperature": max(self.config.temperature, 1e-6),
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                }
            )

        return kwargs

    def _trim_generated_ids(self, token_ids: Tensor) -> Tuple[Tensor, str]:
        ids = token_ids.flatten()
        eos_ids = set(_normalize_eos_ids(self.tokenizer.eos_token_id))
        pad_id = self.tokenizer.pad_token_id

        kept: List[int] = []
        finish_reason = "length"
        for token in ids.detach().cpu().tolist():
            token = int(token)
            if pad_id is not None and token == int(pad_id) and token not in eos_ids:
                finish_reason = "padding"
                break
            kept.append(token)
            if token in eos_ids:
                finish_reason = "eos"
                break

        if not kept:
            return ids[:0], finish_reason

        return torch.tensor(kept, dtype=torch.long, device=ids.device), finish_reason

    def compute_token_log_probs(
        self,
        *,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        generated_token_ids: Tensor,
    ) -> Tensor:
        if prompt_input_ids.dim() != 2 or prompt_input_ids.shape[0] != 1:
            raise ValueError("prompt_input_ids must have shape (1, prompt_length).")
        if prompt_attention_mask.shape != prompt_input_ids.shape:
            raise ValueError(
                "prompt_attention_mask must have the same shape as prompt_input_ids."
            )

        if generated_token_ids.dim() == 1:
            generated_token_ids = generated_token_ids.unsqueeze(0)
        if generated_token_ids.dim() != 2 or generated_token_ids.shape[0] != 1:
            raise ValueError("generated_token_ids must have shape (T,) or (1, T).")

        generated_token_ids = generated_token_ids.to(prompt_input_ids.device)
        num_generated = int(generated_token_ids.shape[-1])
        if num_generated == 0:
            return torch.empty((0,), dtype=torch.float32, device=prompt_input_ids.device)

        prompt_length = int(prompt_input_ids.shape[-1])
        if prompt_length == 0:
            raise ValueError("The controller prompt cannot be empty.")

        completion_attention = torch.ones_like(generated_token_ids, dtype=prompt_attention_mask.dtype)
        full_input_ids = torch.cat([prompt_input_ids, generated_token_ids], dim=1)
        full_attention_mask = torch.cat(
            [prompt_attention_mask, completion_attention],
            dim=1,
        )

        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        start = prompt_length - 1
        end = start + num_generated
        action_logits = outputs.logits[:, start:end, :]

        if int(action_logits.shape[1]) != num_generated:
            raise RuntimeError(
                "Token/logit alignment failure: "
                f"expected {num_generated} action positions, "
                f"got {int(action_logits.shape[1])}."
            )

        log_probs = F.log_softmax(action_logits.float(), dim=-1)
        selected = torch.gather(
            log_probs,
            dim=-1,
            index=generated_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        return selected.squeeze(0)

    def sample_action(
        self,
        *,
        question: str,
        step_id: int,
        evidence_memory: Optional[Any],
        question_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> SampledPolicyAction:
        if step_id < 0:
            raise ValueError("step_id must be non-negative.")

        prompt = self.build_prompt(
            question=question,
            step_id=step_id,
            evidence_memory=evidence_memory,
        )

        with self._rollout_mode(), self._seeded_rng(seed), torch.inference_mode():
            generated = self.model.generate(
                input_ids=prompt.input_ids,
                attention_mask=prompt.attention_mask,
                **self._generation_kwargs(),
            )

            sequences = generated.sequences
            if sequences.dim() != 2 or sequences.shape[0] != 1:
                raise RuntimeError(
                    "sample_action expects exactly one generated sequence; "
                    f"received shape={tuple(sequences.shape)}."
                )

            raw_generated_ids = sequences[0, prompt.num_tokens :]
            generated_ids, finish_reason = self._trim_generated_ids(raw_generated_ids)

            old_log_probs = self.compute_token_log_probs(
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
                generated_token_ids=generated_ids,
            )

        raw_output = self.tokenizer.decode(
            generated_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        format_info = _strict_action_format_diagnostics(raw_output)
        agent_action = self.reasoning_agent.parse_action(
            raw_output=raw_output,
            question=question,
            step_id=step_id,
            evidence_memory=evidence_memory,
        )
        agent_action.raw_output = raw_output
        agent_action.validate()

        action_name = str(agent_action.action)
        if action_name not in ("SearchGraph", "SubmitFinalAnswer"):
            raise RuntimeError(f"ReasoningAgent returned unsupported action {action_name!r}.")

        generated_id_list = generated_ids.detach().cpu().tolist()
        action_mask = [1] * len(generated_id_list)
        eos_ids = set(_normalize_eos_ids(self.tokenizer.eos_token_id))
        if (
            generated_id_list
            and not self.config.include_eos_in_action_mask
            and int(generated_id_list[-1]) in eos_ids
        ):
            action_mask[-1] = 0

        old_log_prob_list = [float(value) for value in old_log_probs.detach().cpu().tolist()]
        if any(not math.isfinite(value) for value in old_log_prob_list):
            raise RuntimeError("Non-finite old policy log-probability was produced.")

        generation_metadata: Dict[str, Any] = {
            "question_id": question_id,
            "seed": seed,
            "model_name_or_path": self.config.model_name_or_path,
            "tokenizer_name_or_path": (
                self.config.tokenizer_name_or_path
                or self.config.model_name_or_path
            ),
            "do_sample": self.config.sampling_enabled,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "max_new_tokens": self.config.max_new_tokens,
            "finish_reason": finish_reason,
            "original_prompt_tokens": prompt.original_num_tokens,
            "effective_prompt_tokens": prompt.num_tokens,
            "prompt_truncated": prompt.truncated,
            "num_generated_tokens": len(generated_id_list),
            "strict_format": format_info,
        }
        if self.config.store_full_prompt_text_in_metadata:
            generation_metadata["full_prompt_text"] = prompt.full_prompt_text

        rollout_action = RolloutAction(
            step_id=step_id,
            prompt_messages=[dict(message) for message in prompt.messages],
            # This is the exact retained prefix after any left truncation.
            prompt_text=prompt.effective_prompt_text,
            prompt_input_ids=prompt.input_ids[0].detach().cpu().tolist(),
            prompt_attention_mask=prompt.attention_mask[0].detach().cpu().tolist(),
            generated_text=raw_output,
            generated_token_ids=generated_id_list,
            old_log_probs=old_log_prob_list,
            action_token_mask=action_mask,
            action_name=action_name,  # type: ignore[arg-type]
            parsed_action=_to_plain(agent_action),
            parse_success=bool(format_info["strict_format_valid"]),
            fallback_used=bool(format_info["fallback_required"]),
            forced_stop=False,
            observation_text="",
            observation_payload=None,
            observation_token_ids=[],
            generation_metadata=generation_metadata,
        )
        rollout_action.validate(require_old_log_probs=True)

        result = SampledPolicyAction(
            agent_action=agent_action,
            rollout_action=rollout_action,
        )
        result.validate()
        return result

    def attach_observation(
        self,
        *,
        rollout_action: RolloutAction,
        observation_text: str,
        observation_payload: Optional[Dict[str, Any]],
        tokenize_observation: bool = False,
    ) -> None:
        """Attach frozen environment feedback after SearchGraph execution.

        Observation tokens are stored only for diagnostics. They are never added
        to action_token_mask and therefore never receive policy-gradient loss.
        The next action's exact prompt already captures their causal effect.
        """

        rollout_action.observation_text = str(observation_text or "")
        rollout_action.observation_payload = _to_plain(observation_payload)

        if tokenize_observation and rollout_action.observation_text:
            ids = self.tokenizer(
                rollout_action.observation_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            rollout_action.observation_token_ids = [int(token) for token in ids]
        else:
            rollout_action.observation_token_ids = []

        rollout_action.validate(require_old_log_probs=True)


__all__ = [
    "GRPOPolicyConfig",
    "PromptEncoding",
    "SampledPolicyAction",
    "GRPOPolicy",
]
