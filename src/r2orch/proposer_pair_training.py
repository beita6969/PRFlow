"""Proposer pair update.

Each candidate list yields two distinct picks a and b with the pair target F~a - F~b.  The
advantage of a is half the target scaled to unit RMS over the batch, b takes its negative, and

L_P = -advantage * (l_phi(a) - l_phi(b)) + beta * KL(pi_phi || pi_0),

where l is the log-softmax over the list's label tokens at the last prompt position, phi is the
Proposer LoRA and 0 the same model under ``disable_adapter()`` (``reinforce_pair_loss``).
Dispatched from ``training.py`` for ``objective == "proposer_pair"``.

Besides the update this module computes the diagnostics that make the Proposer's learning
observable (``pair_signal_stats`` is model-free, ``policy_diagnostics`` needs the label
log-probabilities of the policy and of the reference).  The held-out ones are taken before the
update, on this cycle's own picks, which the policy has not been trained on yet.

Learner-side options (all read with ``getattr`` so bare trainer configs keep working):

* ``proposer_optimizer_steps``: optimizer steps per cycle (0 = ``ppo_mini_batches``).
* ``proposer_max_grad_norm``: gradient-norm cap (0 = ``max_grad_norm``).
* ``proposer_kl_beta``: weight of the label-distribution KL to the reference.
* ``proposer_gap_clip``: trust region on the pair's tilt (0 = off), see ``reinforce_pair_loss``.
"""

from __future__ import annotations

import math
import statistics
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any

LOSS_VARIANT = "proposer_pair"  # the batch objective
_ZERO = 1e-12


def reinforce_pair_loss(
    current: Any, reference: Any, a: int, b: int, advantage: float, kl_beta: float, gap_clip: float = 0.0
):
    """Policy gradient of one list: -advantage * (log pi(a) - log pi(b)) + beta * KL(pi || pi_0).

    ``current`` / ``reference`` are the label log-probabilities (the former differentiable).  The
    two picks are a leave-one-out pair, so their advantages are +advantage and -advantage.
    ``gap_clip`` > 0 is a trust region around the reference: once the pair's tilt
    g = (log pi(a) - log pi(b)) - (log pi_0(a) - log pi_0(b)) is beyond +-gap_clip in the direction the
    advantage pushes, the policy term is dropped (PPO-style clipping of the ratio to pi_0 instead of
    pi_old), so a run of same-signed noisy advantages cannot drive the list's choice deterministic;
    the KL term keeps pulling it back.
    """

    torch_probs = current.exp()
    kl = (torch_probs * (current - reference)).sum()
    gap = (current[a] - current[b]) - (reference[a] - reference[b])
    if gap_clip > 0 and float(advantage) * float(gap.detach()) > 0 and abs(float(gap.detach())) >= gap_clip:
        return float(kl_beta) * kl
    return -float(advantage) * (current[a] - current[b]) + float(kl_beta) * kl


def pair_terms(sample: Any) -> tuple[int, int, float]:
    r2orch = sample.metadata["r2orch"]
    return int(r2orch["choice_a"]["index"]), int(r2orch["choice_b"]["index"]), float(r2orch["target"])


def step_index_chunks(samples: list[Any], count: int) -> list[list[int]]:
    """Sample indices of each optimizer step: contiguous chunks."""

    order = list(range(len(samples)))
    if not order:
        return []
    count = max(1, min(int(count), len(order)))
    width = math.ceil(len(order) / count)
    return [order[start : start + width] for start in range(0, len(order), width)]


# --------------------------------------------------------------------------- statistics


def effective_participation(targets: list[float]) -> float | None:
    """(sum t^2)^2 / sum t^4 over the non-zero targets: how many pairs share a step's gradient.

    1.0 means one pair carries it all, N means N equally large targets.  None without a
    non-zero target (a step that only pulls the policy back to the reference).
    """

    values = [float(value) for value in targets if abs(value) > _ZERO]
    if not values:
        return None
    fourth = sum(value**4 for value in values)
    return sum(value**2 for value in values) ** 2 / fourth


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2.0
        start = stop + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation with average ranks; None when undefined (n < 3 or constant)."""

    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = math.sqrt(sum((value - mx) ** 2 for value in rx))
    sy = math.sqrt(sum((value - my) ** 2 for value in ry))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / (sx * sy)


def pair_signal_stats(samples: list[Any], chunks: list[list[int]]) -> dict[str, Any]:
    """Model-free description of the pair set and of how it is split over optimizer steps."""

    targets = [pair_terms(sample)[2] for sample in samples]
    step_eff, step_top, all_zero = [], [], 0
    for chunk in chunks:
        values = [targets[index] for index in chunk if abs(targets[index]) > _ZERO]
        if not values:
            all_zero += 1
            continue
        step_eff.append(effective_participation(values))
        step_top.append(max(value * value for value in values) / sum(value * value for value in values))
    return {
        "r2orch_pairs_nonzero": sum(1 for value in targets if abs(value) > _ZERO),
        "r2orch_eff_participation": statistics.fmean(step_eff) if step_eff else None,
        "r2orch_eff_participation_all": effective_participation(targets),
        "r2orch_top_pair_share": statistics.fmean(step_top) if step_top else None,
        "r2orch_all_zero_steps": all_zero,
        "r2orch_steps": len(chunks),
    }


def policy_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Held-out and drift diagnostics from pre-update label log-probabilities.

    Each record: ``current`` and ``reference`` (log-softmax over the list's labels), the pair
    indices ``a``/``b``, its ``target`` and, when known, the per-label ``sources`` and
    ``previous_success``.  The ``*_fresh`` values are taken on this cycle's own pairs, before
    the update.
    """

    out: dict[str, Any] = {}
    fresh = records

    def gap(row: dict[str, Any]) -> float:
        a, b = row["a"], row["b"]
        return (row["current"][a] - row["current"][b]) - (row["reference"][a] - row["reference"][b])

    gaps = [gap(row) for row in fresh]
    targets = [row["target"] for row in fresh]
    nonzero = [(g, t) for g, t in zip(gaps, targets, strict=True) if abs(t) > _ZERO]
    out["r2orch_rank_corr_fresh"] = spearman(gaps, targets)
    out["r2orch_rank_corr_nonzero_fresh"] = spearman([g for g, _ in nonzero], [t for _, t in nonzero])
    out["r2orch_sign_agreement_fresh"] = (
        statistics.fmean(1.0 if g * t > 0 else 0.0 for g, t in nonzero) if nonzero else None
    )
    # Sufficient statistics to pool the correlation over cycles (means are ~0 near pi_0).
    out["r2orch_fresh_n"] = len(fresh)
    out["r2orch_fresh_gap_t_sum"] = sum(g * t for g, t in zip(gaps, targets, strict=True))
    out["r2orch_fresh_gap_sq_sum"] = sum(g * g for g in gaps)
    out["r2orch_fresh_t_sq_sum"] = sum(t * t for t in targets)
    kls, entropies, same_expected, p_policy, p_reference, proxy = [], [], [], [], [], []
    for row in records:
        current, reference = row["current"], row["reference"]
        probs = [math.exp(value) for value in current]
        kls.append(sum(p * (c - r) for p, c, r in zip(probs, current, reference, strict=True)))
        entropies.append(-sum(p * c for p, c in zip(probs, current, strict=True)))
        same_expected.append(sum(p * p for p in probs))
        sources = row.get("sources")
        if sources and len(sources) == len(current):
            p_policy.append(sum(p for p, source in zip(probs, sources, strict=True) if source == "ads"))
            p_reference.append(
                sum(math.exp(r) for r, source in zip(reference, sources, strict=True) if source == "ads")
            )
    for row in fresh:
        known = [
            (c - r, float(p) * (1.0 - float(p)))
            for c, r, p in zip(
                row["current"], row["reference"], row.get("previous_success") or [], strict=False
            )
            if p is not None
        ]
        if len(known) >= 4:
            value = spearman([x for x, _ in known], [u for _, u in known])
            if value is not None:
                proxy.append(value)
    out["r2orch_kl_to_ref_mean"] = statistics.fmean(kls) if kls else None
    out["r2orch_kl_to_ref_max"] = max(kls) if kls else None
    out["r2orch_entropy_policy_mean"] = statistics.fmean(entropies) if entropies else None
    out["r2orch_same_choice_expected"] = statistics.fmean(same_expected) if same_expected else None
    out["r2orch_p_ads_policy"] = statistics.fmean(p_policy) if p_policy else None
    out["r2orch_p_ads_ref"] = statistics.fmean(p_reference) if p_reference else None
    out["r2orch_p_ads_drift"] = (
        statistics.fmean(p_policy) - statistics.fmean(p_reference) if p_policy else None
    )
    out["r2orch_proxy_rank_corr"] = statistics.fmean(proxy) if proxy else None
    out["r2orch_proxy_lists"] = len(proxy)
    return out


def _diagnostic_records(samples, currents, references) -> list[dict[str, Any]]:
    records = []
    for sample, current, reference in zip(samples, currents, references, strict=True):
        a, b, target = pair_terms(sample)
        r2orch = sample.metadata["r2orch"]
        records.append(
            {
                "current": [float(value) for value in current],
                "reference": [float(value) for value in reference],
                "a": a,
                "b": b,
                "target": target,
                "sources": r2orch.get("label_sources"),
                "previous_success": r2orch.get("label_previous_success"),
            }
        )
    return records


def _metrics(
    batch: Any,
    losses: list[float],
    gaps: list[float],
    extra: dict[str, Any] | None = None,
):
    targets = [pair_terms(sample)[2] for sample in batch.samples]
    metrics = {
        "r2orch_pair_loss": statistics.fmean(losses),
        "r2orch_pairs": len(batch.samples),
        "r2orch_skipped": len(batch.metadata.get("r2orch_proposer", {}).get("skipped", {})),
        "r2orch_target_abs_mean": statistics.fmean(abs(value) for value in targets),
        "r2orch_logit_gap_mean": statistics.fmean(gaps),
    }
    metrics.update(extra or {})
    return metrics


def _label_log_probs(trainer: Any, sample: Any, *, reference: bool, grad: bool | None = None):
    torch = trainer.torch
    device = trainer.config.device
    support = [int(value) for value in sample.metadata["r2orch"]["label_token_ids"]]
    inputs = torch.tensor([sample.token_ids], dtype=torch.long, device=device)
    position = torch.tensor([len(sample.token_ids) - 1], dtype=torch.long, device=device)
    adapter_context = trainer.model.disable_adapter() if reference else nullcontext()
    differentiable = (not reference) if grad is None else grad
    gradient_context = nullcontext() if differentiable else torch.no_grad()
    with adapter_context, gradient_context:
        logits = trainer._selected_sequence_logits(inputs, torch.ones_like(inputs), position)
        values = torch.log_softmax(logits[0, 0, support].float(), dim=-1)
    return values if differentiable else values.detach()


def proposer_pair_update(trainer: Any, batch: Any, *, step: int):
    from .training import PolicyUpdateResult

    started = time.perf_counter()
    samples = list(batch.samples)
    torch = trainer.torch
    requested_steps = int(
        trainer.config.proposer_optimizer_steps or trainer.config.ppo_mini_batches
    )
    max_grad_norm = float(
        trainer.config.proposer_max_grad_norm or trainer.config.max_grad_norm
    )
    chunks = step_index_chunks(samples, requested_steps)
    kl_beta = float(trainer.config.proposer_kl_beta)
    gap_clip = float(trainer.config.proposer_gap_clip)
    # Unit-RMS advantages: half the pair target, divided by the batch RMS of that half.
    half_targets = [pair_terms(sample)[2] / 2.0 for sample in samples]
    scale = math.sqrt(statistics.fmean(value * value for value in half_targets)) if half_targets else 0.0
    trainer.model.eval()
    references = [_label_log_probs(trainer, sample, reference=True) for sample in samples]
    # Pre-update policy on the same prompts (no grad, no LoRA dropout): the held-out view of
    # fresh pairs and the policy-vs-reference drift, before any step of this cycle.
    currents = [
        _label_log_probs(trainer, sample, reference=False, grad=False) for sample in samples
    ]
    diagnostics = policy_diagnostics(
        _diagnostic_records(
            samples,
            [value.tolist() for value in currents],
            [value.tolist() for value in references],
        )
    )
    del currents
    trainer.model.train()
    losses: list[float] = []
    gaps: list[float] = []
    norms: list[float] = []
    step_metrics: list[dict[str, Any]] = []
    for chunk in chunks:
        window_started = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)
        window_losses = []
        for index in chunk:
            sample = samples[index]
            a, b, target = pair_terms(sample)
            if not math.isfinite(target):
                raise ValueError(
                    f"non-finite Proposer pair target for {sample.rollout_id}: {target!r}"
                )
            current = _label_log_probs(trainer, sample, reference=False)
            reference = references[index].to(current.device)
            gap = (current[a] - current[b]) - (reference[a] - reference[b])
            advantage = (target / 2.0) / scale if scale > _ZERO else 0.0
            loss = reinforce_pair_loss(current, reference, a, b, advantage, kl_beta, gap_clip)
            if not math.isfinite(float(loss.detach())):
                raise ValueError(f"non-finite Proposer pair loss for {sample.rollout_id}")
            (loss / len(chunk)).backward()
            window_losses.append(float(loss.detach()))
            gaps.append(float(gap.detach()))
        norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_grad_norm)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.optimizer_step_count += 1
        losses.extend(window_losses)
        norms.append(float(norm))
        metrics = trainer._step_metrics(
            alternating_update_step=step,
            epoch=0,
            losses=window_losses,
            policy_losses=window_losses,
            kls=[0.0] * len(chunk),
            entropies=[0.0] * len(chunk),
            rewards=[float(samples[index].reward) for index in chunk],
            masked_tokens=len(chunk),
            clipped_tokens=0,
            grad_norm=float(norm),
            duration_seconds=time.perf_counter() - window_started,
            microbatch_sizes=[1] * len(chunk),
        )
        step_targets = [pair_terms(samples[index])[2] for index in chunk]
        metrics.update(
            r2orch_step_pairs=len(chunk),
            r2orch_step_nonzero=sum(1 for value in step_targets if abs(value) > _ZERO),
            r2orch_step_eff_participation=effective_participation(step_targets),
            r2orch_step_grad_norm_pre_clip=float(norm),
            r2orch_step_clipped=bool(float(norm) > max_grad_norm),
        )
        step_metrics.append(metrics)
    extra = {
        **pair_signal_stats(samples, chunks),
        **diagnostics,
        "r2orch_max_grad_norm": max_grad_norm,
        "r2orch_grad_norm_pre_clip_mean": statistics.fmean(norms),
        "r2orch_grad_norm_pre_clip_max": max(norms),
        "r2orch_clip_fraction": sum(1 for value in norms if value > max_grad_norm) / len(norms),
    }
    step_metrics[-1].update(_metrics(batch, losses, gaps, extra))
    checkpoint = trainer._save(step)
    rewards = [float(sample.reward) for sample in samples]
    return PolicyUpdateResult(
        trainer.role,
        step,
        statistics.fmean(losses),
        statistics.fmean(losses),
        0.0,
        len(samples),
        str(checkpoint),
        samples=len(samples),
        optimizer_steps=len(step_metrics),
        learning_rate=float(trainer.scheduler.get_last_lr()[0]),
        grad_norm=statistics.fmean(norms),
        reward_mean=statistics.fmean(rewards),
        reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        duration_seconds=time.perf_counter() - started,
        step_metrics=tuple(step_metrics),
        loss_variant=LOSS_VARIANT,
        reference_provenance="lora_disabled_base",
    )


def mock_proposer_pair_update(trainer: Any, batch: Any, *, step: int):
    """Torch-free stand-in: an untrained policy equals its reference, so gap = 0."""
    from .training import PolicyUpdateResult, _atomic_json

    started = time.perf_counter()
    samples = list(batch.samples)
    # An untrained policy equals its reference: the tilt is 0 and so is the KL, hence the loss.
    losses = [0.0] * len(samples)
    gaps = [0.0] * len(samples)
    destination = trainer.config.checkpoint_root / f"step-{step:08d}"
    destination.mkdir(parents=True, exist_ok=True)
    loss = statistics.fmean(losses)
    _atomic_json(
        destination / "mock_policy.json",
        {"role": trainer.role, "step": step, "loss": loss, "objective": LOSS_VARIANT},
    )
    if trainer.config.publish_latest_checkpoint:
        _atomic_json(
            trainer.config.checkpoint_root / "latest.json",
            {"path": str(destination), "step": step},
        )
    rewards = [float(sample.reward) for sample in samples]
    requested_steps = int(
        trainer.config.proposer_optimizer_steps or trainer.config.ppo_mini_batches
    )
    chunks = step_index_chunks(samples, requested_steps)
    # With the policy equal to its reference every label log-probability is uniform.
    labels = max((len(sample.metadata["r2orch"].get("labels", ())) for sample in samples), default=0)
    uniform = [[-math.log(labels)] * labels for _ in samples] if labels else []
    diagnostics = (
        policy_diagnostics(_diagnostic_records(samples, uniform, uniform)) if labels else {}
    )
    metric = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "role": trainer.role,
        "role_optimizer_step": step,
        "alternating_update_step": step,
        "epoch": 0,
        "microbatch_count": len(samples),
        "loss": loss,
        "policy_loss": loss,
        "kl": 0.0,
        "entropy": 0.0,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "masked_tokens": len(samples),
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "learning_rate": trainer.config.learning_rate,
        "duration_seconds": time.perf_counter() - started,
        **_metrics(
            batch, losses, gaps, {**pair_signal_stats(samples, chunks), **diagnostics}
        ),
    }
    return PolicyUpdateResult(
        trainer.role,
        step,
        loss,
        loss,
        0.0,
        len(samples),
        str(destination),
        samples=len(samples),
        optimizer_steps=len(chunks),
        learning_rate=trainer.config.learning_rate,
        reward_mean=metric["reward_mean"],
        reward_std=metric["reward_std"],
        duration_seconds=metric["duration_seconds"],
        step_metrics=(metric,),
        loss_variant=LOSS_VARIANT,
        reference_provenance="mock",
    )
