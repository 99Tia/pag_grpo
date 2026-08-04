from __future__ import annotations
import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from .schema import OpenIEDoc, Passage, Triple, compute_mdhash_id

logger = logging.getLogger(__name__)

OpenIEBackend = Literal["vllm", "openai", "transformers", "mock"]


@dataclass
class OpenIEExtractorConfig:
    backend: OpenIEBackend = "vllm"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"

    # Generation
    temperature: float = 0.0
    max_ner_tokens: int = 512
    max_triple_tokens: int = 2048

    # OpenAI / OpenAI-compatible endpoint
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None

    # vLLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True

    # Transformers
    device_map: str = "auto"

    # Runtime
    batch_size: int = 8
    save_every: int = 100
    force: bool = False


# Prompt templates
NER_SYSTEM_PROMPT = """You are a precise Open Information Extraction system.
Extract named entities from the passage.
Return only valid JSON.
Do not include explanation.
"""

NER_USER_PROMPT = """Passage:
{passage}

Task:
Extract important named entities and concept phrases needed for knowledge graph construction.

Return JSON exactly in this format:
{{
  "named_entities": ["entity 1", "entity 2"]
}}
"""

TRIPLE_SYSTEM_PROMPT = """You are a precise Open Information Extraction system.
Extract factual relation triples from the passage.
Return only valid JSON.
Do not include explanation.
"""

TRIPLE_USER_PROMPT = """Passage:
{passage}

Named entities:
{named_entities_json}

Task:
Extract factual triples grounded in the passage.

Rules:
- Each triple must be [subject, predicate, object].
- Use concise subject/object strings.
- Use relation predicates that preserve the meaning from the passage.
- Do not invent facts not stated in the passage.
- Prefer triples useful for multi-hop question answering.
- Return only valid JSON.

Return JSON exactly in this format:
{{
  "triples": [
    ["subject", "predicate", "object"]
  ]
}}
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|python)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text

def _find_first_json_object(text: str) -> Optional[str]:
    text = _strip_code_fences(text)

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_json_object(text: str) -> Dict[str, Any]:
    obj_text = _find_first_json_object(text)
    if obj_text is None:
        return {}

    try:
        parsed = json.loads(obj_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(obj_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}

def normalize_entity(entity: Any) -> Optional[str]:
    if entity is None:
        return None

    text = str(entity).strip()
    text = " ".join(text.split())

    if not text:
        return None

    return text

def normalize_entities(entities: Any) -> List[str]:
    if not isinstance(entities, list):
        return []

    result: List[str] = []
    seen = set()

    for ent in entities:
        norm = normalize_entity(ent)
        if norm is None:
            continue

        key = norm.lower()
        if key in seen:
            continue

        result.append(norm)
        seen.add(key)

    return result

def normalize_triple(raw: Any) -> Optional[Tuple[str, str, str]]:
    if not isinstance(raw, (list, tuple)):
        return None

    if len(raw) != 3:
        return None

    subject = normalize_entity(raw[0])
    predicate = normalize_entity(raw[1])
    object_ = normalize_entity(raw[2])

    if subject is None or predicate is None or object_ is None:
        return None

    if subject.lower() == object_.lower():
        return None

    return subject, predicate, object_

def normalize_triples(raw_triples: Any) -> List[Tuple[str, str, str]]:
    if not isinstance(raw_triples, list):
        return []

    triples: List[Tuple[str, str, str]] = []
    seen = set()

    for raw in raw_triples:
        triple = normalize_triple(raw)
        if triple is None:
            continue

        key = tuple(x.lower() for x in triple)
        if key in seen:
            continue

        triples.append(triple)
        seen.add(key)

    return triples


class BaseLLMBackend:
    def generate(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        raise NotImplementedError

    def generate_batch(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
        max_tokens: int,
        desc: str,
    ) -> List[str]:
        outputs = []
        for messages in tqdm(messages_list, desc=desc):
            outputs.append(self.generate(messages, max_tokens=max_tokens))
        return outputs


class MockBackend(BaseLLMBackend):
    def generate(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        user_text = messages[-1]["content"].lower()
        if "named_entities" in user_text:
            return '{"named_entities": []}'
        return '{"triples": []}'


class OpenAIBackend(BaseLLMBackend):
    def __init__(self, config: OpenIEExtractorConfig):
        from openai import OpenAI

        api_key = os.environ.get(config.api_key_env)
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)
        self.model_name = config.model_name
        self.temperature = config.temperature

    def generate(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=max_tokens,
            )

        return response.choices[0].message.content or ""


class VLLMBackend(BaseLLMBackend):
    def __init__(self, config: OpenIEExtractorConfig):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
        )
        self.temperature = config.temperature

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        tokenizer = self.llm.get_tokenizer()

        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}:\n{m['content']}")
            parts.append("ASSISTANT:\n")
            return "\n\n".join(parts)

    def generate(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        prompts = [self._messages_to_prompt(messages)]
        params = self.SamplingParams(
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
        max_tokens: int,
        desc: str,
    ) -> List[str]:
        prompts = [self._messages_to_prompt(messages) for messages in messages_list]
        params = self.SamplingParams(
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(prompts, params)

        results = []
        for output in tqdm(outputs, desc=desc):
            results.append(output.outputs[0].text.strip())
        return results


class TransformersBackend(BaseLLMBackend):
    def __init__(self, config: OpenIEExtractorConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            torch_dtype="auto",
            trust_remote_code=config.trust_remote_code,
        )
        self.temperature = config.temperature

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}:\n{m['content']}")
            parts.append("ASSISTANT:\n")
            return "\n\n".join(parts)

    def generate(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        prompt = self._messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = self.temperature > 0.0

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_backend(config: OpenIEExtractorConfig) -> BaseLLMBackend:
    if config.backend == "mock":
        return MockBackend()

    if config.backend == "openai":
        return OpenAIBackend(config)

    if config.backend == "vllm":
        return VLLMBackend(config)

    if config.backend == "transformers":
        return TransformersBackend(config)

    raise ValueError(f"Unsupported OpenIE backend: {config.backend}")


class OpenIEExtractor:
    def __init__(self, config: OpenIEExtractorConfig):
        self.config = config
        self.backend = build_backend(config)


    def build_ner_messages(self, passage: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": NER_SYSTEM_PROMPT},
            {"role": "user", "content": NER_USER_PROMPT.format(passage=passage)},
        ]

    def build_triple_messages(
        self,
        passage: str,
        named_entities: Sequence[str],
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": TRIPLE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": TRIPLE_USER_PROMPT.format(
                    passage=passage,
                    named_entities_json=json.dumps(
                        {"named_entities": list(named_entities)},
                        ensure_ascii=False,
                    ),
                ),
            },
        ]


    def extract_entities(self, passage: str) -> Tuple[List[str], str]:
        messages = self.build_ner_messages(passage)
        raw_response = self.backend.generate(
            messages,
            max_tokens=self.config.max_ner_tokens,
        )
        parsed = parse_json_object(raw_response)
        entities = normalize_entities(parsed.get("named_entities", []))
        return entities, raw_response

    def extract_triples(
        self,
        passage: str,
        named_entities: Sequence[str],
        source_passage_id: str,
    ) -> Tuple[List[Triple], str]:
        messages = self.build_triple_messages(passage, named_entities)
        raw_response = self.backend.generate(
            messages,
            max_tokens=self.config.max_triple_tokens,
        )
        parsed = parse_json_object(raw_response)
        raw_triples = normalize_triples(parsed.get("triples", []))

        triples = [
            Triple(
                subject=s,
                predicate=p,
                object=o,
                source_passage_id=source_passage_id,
            )
            for s, p, o in raw_triples
        ]

        return triples, raw_response

    def extract_passage(self, passage: Union[str, Passage]) -> OpenIEDoc:
        passage_obj = self._to_passage(passage)

        entities, ner_response = self.extract_entities(passage_obj.text)
        triples, triple_response = self.extract_triples(
            passage=passage_obj.text,
            named_entities=entities,
            source_passage_id=passage_obj.passage_id,
        )

        return OpenIEDoc(
            passage_id=passage_obj.passage_id,
            passage=passage_obj.text,
            extracted_entities=entities,
            extracted_triples=triples,
            metadata={
                "title": passage_obj.title,
                "source_id": passage_obj.source_id,
                "ner_response": ner_response,
                "triple_response": triple_response,
                **passage_obj.metadata,
            },
        )


    def batch_extract(
        self,
        passages: Sequence[Union[str, Passage]],
        output_path: Optional[str] = None,
    ) -> List[OpenIEDoc]:
        
        passage_objs = [self._to_passage(p) for p in passages]

        existing_docs: Dict[str, OpenIEDoc] = {}
        if output_path and os.path.exists(output_path) and not self.config.force:
            for doc in load_openie_results(output_path):
                existing_docs[doc.passage_id] = doc

        to_process = [p for p in passage_objs if p.passage_id not in existing_docs]

        logger.info(
            "OpenIE batch extraction: total=%d, existing=%d, new=%d",
            len(passage_objs),
            len(existing_docs),
            len(to_process),
        )

        new_docs: List[OpenIEDoc] = []

        if to_process:
            # 1) Batch NER
            ner_messages = [self.build_ner_messages(p.text) for p in to_process]
            ner_outputs = self.backend.generate_batch(
                ner_messages,
                max_tokens=self.config.max_ner_tokens,
                desc="OpenIE NER",
            )

            all_entities: List[List[str]] = []
            for raw in ner_outputs:
                parsed = parse_json_object(raw)
                all_entities.append(normalize_entities(parsed.get("named_entities", [])))

            # 2) Batch triples
            triple_messages = [
                self.build_triple_messages(p.text, entities)
                for p, entities in zip(to_process, all_entities)
            ]
            triple_outputs = self.backend.generate_batch(
                triple_messages,
                max_tokens=self.config.max_triple_tokens,
                desc="OpenIE triples",
            )

            for passage_obj, entities, ner_raw, triple_raw in zip(
                to_process,
                all_entities,
                ner_outputs,
                triple_outputs,
            ):
                parsed = parse_json_object(triple_raw)
                raw_triples = normalize_triples(parsed.get("triples", []))

                triples = [
                    Triple(
                        subject=s,
                        predicate=p,
                        object=o,
                        source_passage_id=passage_obj.passage_id,
                    )
                    for s, p, o in raw_triples
                ]

                new_docs.append(
                    OpenIEDoc(
                        passage_id=passage_obj.passage_id,
                        passage=passage_obj.text,
                        extracted_entities=entities,
                        extracted_triples=triples,
                        metadata={
                            "title": passage_obj.title,
                            "source_id": passage_obj.source_id,
                            "ner_response": ner_raw,
                            "triple_response": triple_raw,
                            **passage_obj.metadata,
                        },
                    )
                )

                if output_path and self.config.save_every > 0:
                    total_so_far = len(existing_docs) + len(new_docs)
                    if total_so_far % self.config.save_every == 0:
                        merged = self._merge_docs_in_original_order(
                            passage_objs,
                            existing_docs,
                            new_docs,
                        )
                        save_openie_results(output_path, merged)

        merged_docs = self._merge_docs_in_original_order(
            passage_objs,
            existing_docs,
            new_docs,
        )

        if output_path:
            save_openie_results(output_path, merged_docs)

        return merged_docs

    @staticmethod
    def _merge_docs_in_original_order(
        passage_objs: Sequence[Passage],
        existing_docs: Dict[str, OpenIEDoc],
        new_docs: Sequence[OpenIEDoc],
    ) -> List[OpenIEDoc]:
        new_map = {doc.passage_id: doc for doc in new_docs}

        merged = []
        for passage in passage_objs:
            if passage.passage_id in new_map:
                merged.append(new_map[passage.passage_id])
            elif passage.passage_id in existing_docs:
                merged.append(existing_docs[passage.passage_id])

        return merged

    @staticmethod
    def _to_passage(passage: Union[str, Passage]) -> Passage:
        if isinstance(passage, Passage):
            return passage

        if isinstance(passage, str):
            return Passage.from_text(passage)

        raise TypeError(f"Unsupported passage type: {type(passage).__name__}")


def openie_doc_to_dict(doc: OpenIEDoc) -> Dict[str, Any]:
    triples = [
        [triple.subject, triple.predicate, triple.object]
        for triple in doc.extracted_triples
    ]

    return {
        "idx": doc.passage_id,
        "passage": doc.passage,
        "extracted_entities": list(doc.extracted_entities),
        "extracted_triples": triples,
        "metadata": dict(doc.metadata),
    }


def dict_to_openie_doc(row: Dict[str, Any]) -> OpenIEDoc:
    passage_id = row.get("idx") or row.get("passage_id")
    passage = row.get("passage") or row.get("text") or ""

    if passage_id is None:
        passage_id = compute_mdhash_id(passage, prefix="chunk-")

    raw_triples = row.get("extracted_triples", [])
    normalized = normalize_triples(raw_triples)

    triples = [
        Triple(
            subject=s,
            predicate=p,
            object=o,
            source_passage_id=passage_id,
        )
        for s, p, o in normalized
    ]

    return OpenIEDoc(
        passage_id=passage_id,
        passage=passage,
        extracted_entities=normalize_entities(row.get("extracted_entities", [])),
        extracted_triples=triples,
        metadata=dict(row.get("metadata", {})),
    )


def save_openie_results(path: str, docs: Sequence[OpenIEDoc]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    all_entities = [
        ent
        for doc in docs
        for ent in doc.extracted_entities
    ]

    if all_entities:
        avg_ent_chars = round(sum(len(e) for e in all_entities) / len(all_entities), 4)
        avg_ent_words = round(sum(len(e.split()) for e in all_entities) / len(all_entities), 4)
    else:
        avg_ent_chars = 0.0
        avg_ent_words = 0.0

    payload = {
        "docs": [openie_doc_to_dict(doc) for doc in docs],
        "avg_ent_chars": avg_ent_chars,
        "avg_ent_words": avg_ent_words,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d OpenIE docs to %s", len(docs), path)


def load_openie_results(path: str) -> List[OpenIEDoc]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("docs", payload)
    if not isinstance(rows, list):
        raise ValueError(f"Invalid OpenIE file format: {path}")

    return [dict_to_openie_doc(row) for row in rows]
