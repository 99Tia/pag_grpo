from __future__ import annotations
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
from .schema import compute_mdhash_id

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStoreConfig:
    store_dir: str
    namespace: str
    batch_size: int = 32
    normalize: bool = True
    embedding_dtype: str = "float32"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    _ensure_dir(directory)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    directory = os.path.dirname(path)
    _ensure_dir(directory)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npy", dir=directory)
    os.close(fd)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def _as_list(texts: Sequence[str] | str) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(t) for t in texts]

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / max(norm, eps)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def _batch_iter(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])



class EmbeddingModelAdapter:

    def __init__(self, model: Any):
        self.model = model

    def encode(
        self,
        texts: Sequence[str] | str,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        input_texts = _as_list(texts)

        if self.model is None:
            raise ValueError(
                "Embedding model is None. Provide an embedding model before inserting new strings."
            )

        if hasattr(self.model, "batch_encode"):
            try:
                output = self.model.batch_encode(
                    input_texts,
                    batch_size=batch_size,
                    norm=normalize,
                    **kwargs,
                )
            except TypeError:
                try:
                    output = self.model.batch_encode(input_texts, **kwargs)
                except TypeError:
                    output = self.model.batch_encode(input_texts)

        elif hasattr(self.model, "encode"):
            try:
                output = self.model.encode(
                    input_texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    **kwargs,
                )
            except TypeError:
                try:
                    output = self.model.encode(input_texts, **kwargs)
                except TypeError:
                    output = self.model.encode(input_texts)

        elif callable(self.model):
            try:
                output = self.model(input_texts, **kwargs)
            except TypeError:
                output = self.model(input_texts)

        else:
            raise TypeError(
                "Embedding model must expose batch_encode(), encode(), or be callable."
            )

        array = np.asarray(output, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if normalize:
            array = l2_normalize(array)

        return array.astype(np.float32)


class EmbeddingStore:
    """Lightweight disk-cached embedding store. chunk_store.insert_strings(["passage text"])"""

    def __init__(
        self,
        config: EmbeddingStoreConfig,
        embedding_model: Any = None,
    ):
        self.config = config
        self.namespace = config.namespace.strip("-")
        self.embedding_model = EmbeddingModelAdapter(embedding_model)

        _ensure_dir(self.config.store_dir)

        self.records_path = os.path.join(self.config.store_dir, "records.jsonl")
        self.embeddings_path = os.path.join(self.config.store_dir, "embeddings.npy")

        self.hash_ids: List[str] = []
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        self.hash_id_to_idx: Dict[str, int] = {}
        self.text_to_hash_id: Dict[str, str] = {}

        self._load()


    def make_hash_id(self, text: str) -> str:
        return compute_mdhash_id(str(text), prefix=f"{self.namespace}-")

    def get_hash_id(self, text: str) -> str:
        return self.text_to_hash_id[str(text)]

    def has_text(self, text: str) -> bool:
        return str(text) in self.text_to_hash_id

    def has_id(self, hash_id: str) -> bool:
        return hash_id in self.hash_id_to_idx


    def _load(self) -> None:
        if os.path.exists(self.records_path):
            with open(self.records_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.hash_ids.append(row["hash_id"])
                    self.texts.append(row["content"])
                    self.metadata.append(row.get("metadata", {}))

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path).astype(np.float32)

        if len(self.hash_ids) == 0:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)

        if len(self.hash_ids) != len(self.texts):
            raise ValueError(
                f"Corrupt embedding store: hash_ids={len(self.hash_ids)}, texts={len(self.texts)}"
            )

        if len(self.hash_ids) > 0 and len(self.embeddings) != len(self.hash_ids):
            raise ValueError(
                f"Corrupt embedding store: records={len(self.hash_ids)}, "
                f"embeddings={len(self.embeddings)}"
            )

        self._rebuild_maps()

        if self.hash_ids:
            logger.info(
                "Loaded %d %s embeddings from %s",
                len(self.hash_ids),
                self.namespace,
                self.config.store_dir,
            )

    def _rebuild_maps(self) -> None:
        self.hash_id_to_idx = {h: i for i, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}

    def save(self) -> None:
        rows = []
        for hash_id, text, meta in zip(self.hash_ids, self.texts, self.metadata):
            rows.append(
                json.dumps(
                    {
                        "hash_id": hash_id,
                        "content": text,
                        "metadata": meta,
                    },
                    ensure_ascii=False,
                )
            )

        _atomic_write_text(self.records_path, "\n".join(rows) + ("\n" if rows else ""))

        if self.embeddings.size == 0:
            empty = np.zeros((0, 0), dtype=np.float32)
            _atomic_save_npy(self.embeddings_path, empty)
        else:
            _atomic_save_npy(
                self.embeddings_path,
                self.embeddings.astype(self.config.embedding_dtype),
            )

        logger.info(
            "Saved %d %s embeddings to %s",
            len(self.hash_ids),
            self.namespace,
            self.config.store_dir,
        )


    def get_missing_string_hash_ids(self, texts: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}

        for text in _as_list(texts):
            hash_id = self.make_hash_id(text)
            if hash_id not in self.hash_id_to_idx:
                result[hash_id] = {
                    "hash_id": hash_id,
                    "content": text,
                }

        return result

    def insert_strings(
        self,
        texts: Sequence[str],
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        input_texts = _as_list(texts)

        if metadata is None:
            input_metadata = [{} for _ in input_texts]
        else:
            input_metadata = list(metadata)
            if len(input_metadata) != len(input_texts):
                raise ValueError("metadata length must match texts length.")

        all_input_ids = [self.make_hash_id(text) for text in input_texts]

        missing_texts: List[str] = []
        missing_ids: List[str] = []
        missing_metadata: List[Dict[str, Any]] = []

        seen_new_ids: Set[str] = set()

        for text, hash_id, meta in zip(input_texts, all_input_ids, input_metadata):
            if hash_id in self.hash_id_to_idx:
                continue
            if hash_id in seen_new_ids:
                continue

            missing_texts.append(text)
            missing_ids.append(hash_id)
            missing_metadata.append(meta)
            seen_new_ids.add(hash_id)

        logger.info(
            "[%s] input=%d, missing=%d, existing=%d",
            self.namespace,
            len(input_texts),
            len(missing_texts),
            len(input_texts) - len(missing_texts),
        )

        if not missing_texts:
            return all_input_ids

        encode_kwargs = encode_kwargs or {}

        new_embeddings_list: List[np.ndarray] = []

        for batch in _batch_iter(missing_texts, self.config.batch_size):
            batch_embeddings = self.embedding_model.encode(
                batch,
                batch_size=self.config.batch_size,
                normalize=self.config.normalize,
                **encode_kwargs,
            )
            new_embeddings_list.append(batch_embeddings)

        new_embeddings = np.vstack(new_embeddings_list).astype(self.config.embedding_dtype)

        self._append_records(missing_ids, missing_texts, missing_metadata, new_embeddings)
        self.save()

        return all_input_ids

    def _append_records(
        self,
        hash_ids: Sequence[str],
        texts: Sequence[str],
        metadata: Sequence[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        if len(hash_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Number of hash IDs ({len(hash_ids)}) does not match embeddings ({embeddings.shape[0]})."
            )

        if self.embeddings.size == 0:
            self.embeddings = embeddings.astype(self.config.embedding_dtype)
        else:
            if self.embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: existing={self.embeddings.shape[1]}, "
                    f"new={embeddings.shape[1]}"
                )
            self.embeddings = np.vstack(
                [self.embeddings, embeddings.astype(self.config.embedding_dtype)]
            )

        self.hash_ids.extend(list(hash_ids))
        self.texts.extend(list(texts))
        self.metadata.extend([dict(m) for m in metadata])

        self._rebuild_maps()

    def delete(self, hash_ids: Sequence[str]) -> None:
        ids_to_delete = set(hash_ids)
        if not ids_to_delete:
            return

        keep_indices = [
            i for i, hash_id in enumerate(self.hash_ids) if hash_id not in ids_to_delete
        ]

        self.hash_ids = [self.hash_ids[i] for i in keep_indices]
        self.texts = [self.texts[i] for i in keep_indices]
        self.metadata = [self.metadata[i] for i in keep_indices]

        if self.embeddings.size == 0:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            self.embeddings = self.embeddings[keep_indices]

        self._rebuild_maps()
        self.save()


    def get_row(self, hash_id: str) -> Dict[str, Any]:
        idx = self.hash_id_to_idx[hash_id]
        return {
            "hash_id": self.hash_ids[idx],
            "content": self.texts[idx],
            "metadata": self.metadata[idx],
        }

    def get_rows(self, hash_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        return {hash_id: self.get_row(hash_id) for hash_id in hash_ids}

    def get_all_ids(self) -> List[str]:
        return list(self.hash_ids)

    def get_all_texts(self) -> Set[str]:
        return set(self.texts)

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, Any]]:
        return {hash_id: self.get_row(hash_id) for hash_id in self.hash_ids}

    def get_embedding(self, hash_id: str, dtype: Any = np.float32) -> np.ndarray:
        idx = self.hash_id_to_idx[hash_id]
        return self.embeddings[idx].astype(dtype)

    def get_embeddings(
        self,
        hash_ids: Optional[Sequence[str]] = None,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        if hash_ids is None:
            return self.embeddings.astype(dtype)

        if not hash_ids:
            return np.zeros((0, self.embedding_dim()), dtype=dtype)

        indices = [self.hash_id_to_idx[h] for h in hash_ids]
        return self.embeddings[indices].astype(dtype)

    def embedding_dim(self) -> int:
        if self.embeddings.size == 0:
            return 0
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.hash_ids)


def make_embedding_store(
    root_dir: str,
    namespace: str,
    embedding_model: Any = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> EmbeddingStore:
    namespace = namespace.strip("-")
    store_dir = os.path.join(root_dir, f"{namespace}_embeddings")

    return EmbeddingStore(
        EmbeddingStoreConfig(
            store_dir=store_dir,
            namespace=namespace,
            batch_size=batch_size,
            normalize=normalize,
        ),
        embedding_model=embedding_model,
    )


def make_default_embedding_stores(
    root_dir: str,
    embedding_model: Any = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> Dict[str, EmbeddingStore]:
    return {
        "chunk": make_embedding_store(
            root_dir=root_dir,
            namespace="chunk",
            embedding_model=embedding_model,
            batch_size=batch_size,
            normalize=normalize,
        ),
        "entity": make_embedding_store(
            root_dir=root_dir,
            namespace="entity",
            embedding_model=embedding_model,
            batch_size=batch_size,
            normalize=normalize,
        ),
        "fact": make_embedding_store(
            root_dir=root_dir,
            namespace="fact",
            embedding_model=embedding_model,
            batch_size=batch_size,
            normalize=normalize,
        ),
    }