from __future__ import annotations
import argparse
import hashlib
import logging
import os
import sys
import json
from urllib import error as urllib_error
from urllib import request as urllib_request
from typing import Any, List, Optional, Sequence
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ppr_agent.triple_index import TripleIndex, TripleIndexConfig  # noqa: E402


logger = logging.getLogger(__name__)


class MockEmbeddingModel:

    def __init__(self, dim: int = 128):
        self.dim = dim

    def batch_encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        return self.encode(texts, **kwargs)

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        vectors = []

        for text in texts:
            seed = int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            vectors.append(vec)

        return np.vstack(vectors).astype(np.float32)


class SentenceTransformerEmbeddingModel:

    def __init__(self, model_name: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def batch_encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        norm: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=norm,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.batch_encode(
            texts,
            batch_size=batch_size,
            norm=normalize_embeddings,
            **kwargs,
        )


class RemoteEmbeddingModel:

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        model_name: Optional[str] = None,
    ):
        base_url = str(base_url or "").strip().rstrip("/")
        if not base_url:
            raise ValueError(
                "Remote embedding backend requires --embedding_base_url."
            )

        self.base_url = base_url
        self.timeout = float(timeout)
        self.model_name = model_name

    def batch_encode(
        self,
        texts: Sequence[str],
        batch_size: int = 4,
        norm: bool = True,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if "normalize" in kwargs:
            norm = bool(kwargs.pop("normalize"))
        if "normalize_embeddings" in kwargs:
            norm = bool(kwargs.pop("normalize_embeddings"))
        texts = [str(text) for text in texts]

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        outputs: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            emb = self._request_embeddings(
                texts=batch,
                normalize=norm,
                instruction=instruction,
            )
            outputs.append(emb)

            logger.info(
                "Remote-encoded %d/%d texts",
                min(start + batch_size, len(texts)),
                len(texts),
            )

        return np.vstack(outputs).astype(np.float32)

    def encode(self, texts: Sequence[str], batch_size: int = 4, normalize_embeddings: bool = True, instruction: Optional[str] = None, **kwargs: Any,) -> np.ndarray:
        normalize = kwargs.pop(
            "normalize",
            normalize_embeddings,
        )
        return self.batch_encode(
            texts=texts,
            batch_size=batch_size,
            norm=bool(normalize),
            instruction=instruction,
            **kwargs,
        )

    def _request_embeddings(
        self,
        texts: Sequence[str],
        normalize: bool,
        instruction: Optional[str],
    ) -> np.ndarray:
        payload = {
            "texts": list(texts),
            "instruction": instruction,
            "normalize": bool(normalize),
        }

        if self.model_name:
            payload["model"] = self.model_name

        request_body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url=f"{self.base_url}/embed",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_payload = json.loads(
                    response.read().decode("utf-8")
                )
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding service returned HTTP {exc.code}: {body}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                f"Could not reach embedding service at {self.base_url}: "
                f"{exc.reason}"
            ) from exc

        embeddings = response_payload.get("embeddings")
        if embeddings is None:
            raise RuntimeError(
                "Embedding service response is missing 'embeddings'."
            )

        array = np.asarray(embeddings, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise RuntimeError(
                "Embedding service returned an invalid array shape: "
                f"{array.shape}"
            )

        if array.shape[0] != len(texts):
            raise RuntimeError(
                "Embedding service returned a different number of vectors: "
                f"expected {len(texts)}, received {array.shape[0]}."
            )

        return array


class HFEmbeddingModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        trust_remote_code: bool = True,
        max_length: int = 4096,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )

        self.model.to(device)
        self.model.eval()

    def batch_encode(
        self,
        texts: Sequence[str],
        batch_size: int = 4,
        norm: bool = True,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        texts = list(texts)
        outputs: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            emb = self._encode_one_batch(
                batch,
                normalize=norm,
                instruction=instruction,
            )
            outputs.append(emb)

            logger.info("Encoded %d/%d texts", min(start + batch_size, len(texts)), len(texts))

        return np.vstack(outputs).astype(np.float32)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 4,
        normalize_embeddings: bool = True,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.batch_encode(
            texts,
            batch_size=batch_size,
            norm=normalize_embeddings,
            instruction=instruction,
            **kwargs,
        )

    def _encode_one_batch(
        self,
        texts: Sequence[str],
        normalize: bool = True,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        # NV-Embed-v2 remote-code path.
        if hasattr(self.model, "encode"):
            try:
                with self.torch.no_grad():
                    if instruction is not None:
                        emb = self.model.encode(
                            list(texts),
                            instruction=instruction,
                            max_length=self.max_length,
                        )
                    else:
                        emb = self.model.encode(
                            list(texts),
                            max_length=self.max_length,
                        )

                emb = self._to_numpy(emb)

                if normalize:
                    emb = self._l2_normalize(emb)

                return emb.astype(np.float32)

            except Exception as exc:
                logger.warning(
                    "model.encode() failed; falling back to generic forward. Error: %s",
                    exc,
                )

        # Generic encoder fallback.
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            emb = summed / counts
        elif isinstance(outputs, tuple):
            emb = outputs[0][:, 0]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            raise RuntimeError(
                f"Could not extract embeddings from HF model output. "
                f"Output type: {type(outputs)}"
            )

        emb = self._to_numpy(emb)

        if normalize:
            emb = self._l2_normalize(emb)

        return emb.astype(np.float32)
    

    def _to_numpy(self, emb: Any) -> np.ndarray:
        if isinstance(emb, np.ndarray):
            arr = emb

        elif self.torch.is_tensor(emb):
            arr = emb.detach().float().cpu().numpy()

        elif isinstance(emb, (list, tuple)):
            if len(emb) > 0 and self.torch.is_tensor(emb[0]):
                arr = self.torch.stack(list(emb)).detach().float().cpu().numpy()
            else:
                arr = np.asarray(emb, dtype=np.float32)

        elif isinstance(emb, dict):
            for key in ["embeddings", "sentence_embeddings", "last_hidden_state", "pooler_output"]:
                if key in emb:
                    return self._to_numpy(emb[key])
            raise RuntimeError(f"Unsupported embedding dict keys: {list(emb.keys())}")

        else:
            arr = np.asarray(emb, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return x / norms


def build_embedding_model(args: argparse.Namespace) -> Any:
    if args.embedding_backend == "mock":
        return MockEmbeddingModel(dim=args.mock_dim)

    if args.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbeddingModel(
            model_name=args.embedding_model_name,
            device=args.device,
        )

    if args.embedding_backend == "hf":
        return HFEmbeddingModel(
            model_name=args.embedding_model_name,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_length=args.max_length,
        )

    if args.embedding_backend == "remote":
        return RemoteEmbeddingModel(
            base_url=getattr(args, "embedding_base_url", None),
            timeout=getattr(args, "embedding_timeout", 300.0),
            model_name=getattr(args, "embedding_model_name", None),
        )

    raise ValueError(f"Unknown embedding backend: {args.embedding_backend}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build triple/passage/entity embedding index."
    )

    parser.add_argument(
        "--graph_metadata_path",
        type=str,
        required=True,
        help="Path to graph_metadata.json from build_agent_graph.py.",
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Output index directory.",
    )

    parser.add_argument(
        "--embedding_backend",
        type=str,
        default="hf",
        choices=["mock", "hf", "sentence_transformers", "remote"],
        help="Embedding backend.",
    )

    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="nvidia/NV-Embed-v2",
        help="Embedding model name or local path.",
    )

    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default=None,
        help=(
            "Base URL for the remote embedding service, for example "
            "http://localhost:8003."
        ),
    )
    parser.add_argument(
        "--embedding_timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for one remote embedding request.",
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--mock_dim", type=int, default=128)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild index metadata. Embedding store still reuses cached embeddings.",
    )

    parser.add_argument(
        "--query_to_fact_instruction",
        type=str,
        default=None,
        help="Optional query instruction for query-to-fact embedding.",
    )

    parser.add_argument(
        "--query_to_passage_instruction",
        type=str,
        default=None,
        help="Optional query instruction for query-to-passage embedding.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not os.path.exists(args.graph_metadata_path):
        raise FileNotFoundError(f"graph_metadata_path not found: {args.graph_metadata_path}")

    os.makedirs(args.index_dir, exist_ok=True)

    logger.info("Building embedding model.")
    logger.info("Embedding backend: %s", args.embedding_backend)
    logger.info("Embedding model: %s", args.embedding_model_name)

    embedding_model = build_embedding_model(args)

    config = TripleIndexConfig(
        index_dir=args.index_dir,
        graph_metadata_path=args.graph_metadata_path,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        query_to_fact_instruction=args.query_to_fact_instruction,
        query_to_passage_instruction=args.query_to_passage_instruction,
    )

    triple_index = TripleIndex(
        config=config,
        embedding_model=embedding_model,
    )

    result = triple_index.build(force=args.force)

    print("\nTriple index build result:")
    print(f"index_dir: {result.index_dir}")
    print(f"num_passages: {result.num_passages}")
    print(f"num_entities: {result.num_entities}")
    print(f"num_triples: {result.num_triples}")
    print(f"metadata_path: {result.metadata_path}")

    print("\nSaved stores:")
    print(os.path.join(args.index_dir, "chunk_embeddings"))
    print(os.path.join(args.index_dir, "entity_embeddings"))
    print(os.path.join(args.index_dir, "fact_embeddings"))


if __name__ == "__main__":
    main()
