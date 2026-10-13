# Copyright 2026 ADS authors
# SPDX-License-Identifier: Apache-2.0
"""ADS-compatible preprocessing for the project's trusted JSONL task pools.

The mean pooling, base-policy NLL, PCA, K-Means, and within-cluster sorting
stages are migrated from ADS.  Unlike ADS's VeRL parquet schema, this module
writes the resulting fields back into the project's JSONL task contract so the
online scheduler and TSDS retriever can consume them.
"""

from __future__ import annotations

import gc
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ADSPreprocessingConfig:
    model_path: Path
    output_path: Path
    artifacts_dir: Path
    num_clusters: int
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    batch_size: int = 1
    max_length: int = 4096
    pca_dim: int = 128
    seed: int = 42
    overwrite: bool = False

    def validate(self, sample_count: int) -> None:
        if sample_count < 3:
            raise ValueError("ADS preprocessing requires at least three training rows")
        if not 2 <= self.num_clusters < sample_count:
            raise ValueError(
                f"num_clusters must be in [2, {sample_count - 1}], got {self.num_clusters}"
            )
        if min(self.batch_size, self.max_length, self.pca_dim) <= 0:
            raise ValueError("batch_size, max_length, and pca_dim must be positive")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"output already exists: {self.output_path}; pass --overwrite to replace it"
            )


def load_ads_input_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    occurrences: Counter[str] = Counter()
    for source in paths:
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                if str(row.get("split", "train")).casefold() != "train":
                    raise ValueError(
                        f"ADS preprocessing accepts train rows only: {path}:{line_number}"
                    )
                prompt = str(row.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError(f"row has no prompt: {path}:{line_number}")
                source_id = str(row.get("id", f"{path.name}:{line_number}"))
                duplicate_index = occurrences[source_id]
                occurrences[source_id] += 1
                rows.append(
                    {
                        **row,
                        "_ads_sample_id": (
                            source_id
                            if duplicate_index == 0
                            else f"{source_id}#row-{duplicate_index}"
                        ),
                    }
                )
    if not rows:
        raise ValueError("ADS input pool is empty")
    return rows


def ads_target_text(row: dict[str, Any]) -> tuple[str, str]:
    """Return the trusted target used by ADS NLL, with explicit provenance."""

    explicit = row.get("ads_target")
    if _has_target_value(explicit):
        return _target_string(explicit), "ads_target"

    metadata = dict(row.get("metadata") or {})
    dataset = str(row.get("dataset", "")).casefold()
    if dataset == "aime" and _has_target_value(metadata.get("solutions")):
        solutions = metadata["solutions"]
        value = solutions[0] if isinstance(solutions, list) else solutions
        source = "metadata.solutions[0]"
    elif dataset == "alfworld":
        value = metadata.get("high_level_descriptions")
        source = "metadata.high_level_descriptions"
    else:
        value = None
        source = ""
    if _has_target_value(value):
        return _target_string(value), source

    answers = row.get("target_answers")
    if isinstance(answers, list) and answers:
        return _target_string(answers[0]), "target_answers[0]"

    raise ValueError(
        f"{row.get('id', '<unknown>')} has no trusted ADS target; "
        "provide top-level ads_target instead of using file order as difficulty"
    )


def _has_target_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value not in (None, [], {})


def _target_string(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value).strip()
    if not text:
        raise ValueError("ADS target is empty")
    return text


def _sequence_text(row: dict[str, Any]) -> tuple[str, str, str]:
    target, source = ads_target_text(row)
    prompt_text = f"Question:\n{str(row['prompt']).strip()}\n\nAnswer:\n"
    return prompt_text, target, source


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("base-policy tokenizer has no pad or eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def extract_base_policy_embeddings(
    rows: Sequence[dict[str, Any]], config: ADSPreprocessingConfig
) -> np.ndarray:
    """Mean-pool final-layer base-policy states, following ADS phase 1."""

    import torch
    from transformers import AutoModel

    tokenizer = _load_tokenizer(config.model_path)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        dtype=_torch_dtype(config.dtype),
        trust_remote_code=True,
    )
    model.eval().to(config.device)
    chunks: list[np.ndarray] = []
    try:
        for start in range(0, len(rows), config.batch_size):
            texts = []
            for row in rows[start : start + config.batch_size]:
                prompt, target, _source = _sequence_text(row)
                texts.append(prompt + target)
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(config.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            chunks.append(pooled.detach().float().cpu().numpy())
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _mean_token_nll(logits: Any, targets: Any, *, chunk_size: int = 256) -> float:
    import torch.nn.functional as functional

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if logits.shape[0] != targets.shape[0] or targets.shape[0] == 0:
        raise ValueError("logits and targets must contain the same nonzero token count")
    nll_sum = 0.0
    for start in range(0, targets.shape[0], chunk_size):
        end = min(start + chunk_size, targets.shape[0])
        chunk_nll = functional.cross_entropy(
            logits[start:end].float(), targets[start:end], reduction="sum"
        )
        nll_sum += float(chunk_nll.item())
    return nll_sum / targets.shape[0]


def compute_base_policy_nll(
    rows: Sequence[dict[str, Any]], config: ADSPreprocessingConfig
) -> tuple[np.ndarray, list[str]]:
    """Compute mean target-token NLL, following ADS phase 4."""

    import torch

    from .qwen_compat import load_training_model

    tokenizer = _load_tokenizer(config.model_path)
    model = load_training_model(config.model_path, dtype=_torch_dtype(config.dtype))
    model.eval().to(config.device)
    scores: list[float] = []
    sources: list[str] = []
    try:
        for row in rows:
            prompt, target, source = _sequence_text(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(prompt + target, add_special_tokens=False)
            full_ids = full_ids[: config.max_length]
            prompt_length = len(prompt_ids)
            answer_length = len(full_ids) - prompt_length
            if prompt_length < 1 or answer_length <= 0:
                scores.append(float("inf"))
                sources.append(source)
                continue
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=config.device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, use_cache=False).logits
            answer_logits = logits[0, prompt_length - 1 : prompt_length + answer_length - 1, :]
            answer_targets = input_ids[0, prompt_length : prompt_length + answer_length]
            # Avoid a full float32 [tokens, vocabulary] copy for long targets.
            scores.append(_mean_token_nll(answer_logits, answer_targets))
            sources.append(source)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.asarray(scores, dtype=np.float64), sources


def build_ads_records(
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    difficulties: np.ndarray,
    target_sources: Sequence[str],
    config: ADSPreprocessingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    """Apply ADS PCA/K-Means and easy-to-hard sorting to JSONL rows."""

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    count = len(rows)
    config.validate(count)
    if embeddings.shape[0] != count or embeddings.ndim != 2:
        raise ValueError("embedding matrix does not match ADS rows")
    if difficulties.shape != (count,) or len(target_sources) != count:
        raise ValueError("difficulty output does not match ADS rows")
    if not np.isfinite(embeddings).all():
        raise ValueError("base-policy embeddings contain NaN or Inf")

    finite = np.isfinite(difficulties)
    if not finite.any():
        raise ValueError("all ADS difficulty scores are non-finite")
    normalized_difficulties = difficulties.copy()
    normalized_difficulties[~finite] = float(np.max(difficulties[finite]) + 1.0)

    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
    actual_pca_dim = min(config.pca_dim, embeddings.shape[1], count - 1)
    if actual_pca_dim < 2:
        raise ValueError("ADS PCA needs at least two dimensions")
    pca = PCA(
        n_components=actual_pca_dim,
        random_state=config.seed,
        svd_solver="randomized",
    )
    features = pca.fit_transform(normalized).astype(np.float32, copy=False)
    clusterer = KMeans(
        n_clusters=config.num_clusters,
        random_state=config.seed,
        n_init="auto",
    )
    labels = clusterer.fit_predict(features).astype(np.int32)

    datasets = sorted({str(row.get("dataset", "unknown")) for row in rows})
    namespace = "+".join(datasets)
    positions_by_cluster: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        positions_by_cluster.setdefault(int(label), []).append(index)
    ranks: dict[int, int] = {}
    cluster_sizes: dict[int, int] = {}
    for _label, positions in positions_by_cluster.items():
        ordered = sorted(positions, key=lambda index: (normalized_difficulties[index], index))
        for rank, index in enumerate(ordered):
            ranks[index] = rank
            cluster_sizes[index] = len(ordered)

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        metadata = dict(row.get("metadata") or {})
        label = int(labels[index])
        record = {key: value for key, value in row.items() if key != "_ads_sample_id"}
        record.update(
            {
                "ads_sample_id": str(row["_ads_sample_id"]),
                "embedding": [float(value) for value in features[index]],
                "cluster_id": f"{namespace}:ads:{label}",
                "difficulty_score": float(normalized_difficulties[index]),
                "rank_in_cluster": int(ranks[index]),
                "cluster_size": int(cluster_sizes[index]),
                "metadata": {
                    **metadata,
                    "ads_preprocessing": {
                        "base_model_path": str(config.model_path),
                        "embedding": "mean_final_hidden_question_plus_target",
                        "embedding_dim": int(embeddings.shape[1]),
                        "pca_dim": int(actual_pca_dim),
                        "clustering": "kmeans",
                        "num_clusters": int(config.num_clusters),
                        "difficulty": "mean_target_token_nll",
                        "target_source": str(target_sources[index]),
                        "seed": int(config.seed),
                    },
                },
            }
        )
        # Explicit ADS targets are offline-only supervision. Dataset runtimes
        # either use their trusted verifier payload or own environment state.
        record.pop("ads_target", None)
        records.append(record)
    records.sort(key=lambda row: (str(row["cluster_id"]), int(row["rank_in_cluster"])))

    cluster_counts = Counter(str(record["cluster_id"]) for record in records)
    manifest = {
        "schema": "ads_preprocessed_fixed_pool",
        "sample_count": count,
        "datasets": datasets,
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
            "output_path": str(config.output_path),
            "artifacts_dir": str(config.artifacts_dir),
        },
        "embedding": {
            "raw_dimension": int(embeddings.shape[1]),
            "pca_dimension": int(actual_pca_dim),
        },
        "clusters": dict(sorted(cluster_counts.items())),
        "difficulty": {
            "minimum": float(np.min(normalized_difficulties)),
            "maximum": float(np.max(normalized_difficulties)),
            "mean": float(np.mean(normalized_difficulties)),
            "replaced_non_finite": int(np.sum(~finite)),
            "target_sources": dict(Counter(target_sources)),
        },
    }
    return records, manifest, features, labels


def prepare_ads_pool(
    input_paths: Iterable[str | Path], config: ADSPreprocessingConfig
) -> dict[str, Any]:
    rows = load_ads_input_rows(input_paths)
    config.validate(len(rows))
    for row in rows:
        ads_target_text(row)

    embeddings = extract_base_policy_embeddings(rows, config)
    difficulties, target_sources = compute_base_policy_nll(rows, config)
    records, manifest, features, labels = build_ads_records(
        rows, embeddings, difficulties, target_sources, config
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.artifacts_dir / "base_policy_mean_embeddings.npy", embeddings)
    np.save(config.artifacts_dir / "pca_cluster_features.npy", features)
    np.save(config.artifacts_dir / "difficulty_nll.npy", difficulties)
    np.save(config.artifacts_dir / "cluster_labels.npy", labels)
    (config.artifacts_dir / "sample_ids.json").write_text(
        json.dumps([str(row["_ads_sample_id"]) for row in rows], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    config.output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (config.artifacts_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
