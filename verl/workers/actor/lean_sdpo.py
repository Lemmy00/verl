"""Lean integration checks and diagnostics around the pinned upstream SDPO actor.

The forward, optimizer, distillation and EMA calculations live in sdpo_upstream.py.
An optional local hybrid term reproduces verified answers under the student prompt;
failed attempts keep the pinned SDPO objective. This term is not upstream SDPO.
"""

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.fsdp_utils import fsdp_version

from verl.workers.actor.sdpo_upstream import DataParallelPPOActor


def compute_verified_self_imitation_loss(student_log_probs, response_mask, weights):
    """Average each response's NLL, then apply precomputed per-theorem weights.

    For a group of n attempts with k eligible verified responses, each verified
    row has weight n/k. Averaging over all n rows therefore gives the mean CE of
    that theorem's k verified answers, independent of the number of successes.
    The caller supplies the full response mask, including EOS and excluding pads.
    This is supervised imitation, so SDPO importance ratios do not apply.
    """
    log_probs = student_log_probs.float()
    mask = response_mask.to(log_probs.dtype)
    per_sequence = -(log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return (per_sequence * weights.to(log_probs.dtype)).mean()


class LeanSDPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        sdpo = config.self_distillation
        if config.use_kl_loss:
            raise ValueError("Upstream SDPO requires use_kl_loss=false")
        if sdpo.teacher_regularization != "ema":
            raise ValueError("The Lean upstream adapter currently supports the EMA recipe only")
        if sdpo.loss_coef != 1.0 or sdpo.distill_proof_only:
            raise ValueError("Upstream SDPO requires loss_coef=1 and full-response distillation")
        if config.get("feedback_loss", {}).get("enabled", False):
            raise ValueError("Upstream SDPO does not implement the local feedback CE objective")
        if config.get("sft_replay", {}).get("enabled", False):
            raise ValueError("Upstream SDPO does not implement the local SFT replay objective")
        super().__init__(config, actor_module, actor_optimizer)
        self._teacher_ema_updates = 0
        self._teacher_sample_delta = 0.0

    def _sdpo_extra_batch_keys(self):
        if self.config.self_distillation.verified_self_imitation_coef > 0:
            return ["verified_self_imitation_weights"]
        return []

    def _compute_sdpo_loss(self, model_inputs, **kwargs):
        sdpo_loss, metrics = super()._compute_sdpo_loss(model_inputs, **kwargs)
        coef = self.config.self_distillation.verified_self_imitation_coef
        if coef == 0:
            return sdpo_loss, metrics
        weights = model_inputs["verified_self_imitation_weights"]
        if torch.any((weights > 0) & (kwargs["self_distillation_mask"] > 0)):
            raise ValueError("Verified self-imitation and repair SDPO must have disjoint rows")
        ce_loss = compute_verified_self_imitation_loss(
            kwargs["student_log_probs"], kwargs["response_mask"], weights
        )
        metrics.update({
            "self_distillation/repair_loss": sdpo_loss.detach().item(),
            "self_distillation/verified_ce": ce_loss.detach().item(),
            "self_distillation/verified_ce_scaled": (coef * ce_loss).detach().item(),
            "self_distillation/verified_ce_coef": coef,
            "self_distillation/verified_ce_weight_mean": weights.float().mean().item(),
        })
        return sdpo_loss + coef * ce_loss, metrics

    @torch.no_grad()
    def _update_teacher(self):
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise RuntimeError("SDPO teacher must be a separate model before any update")
        # In this PyTorch/FSDP runtime a no-grad root forward can retain its full
        # parameter buffer. Invalidate it before changing the shards, as the local
        # reference-logprob worker does. Otherwise a subsequent forward or save
        # may use the pre-EMA weights even though parameters() shows new shards.
        if isinstance(self.teacher_module, FSDP):
            if self.teacher_module._handle is not None:
                self.teacher_module._handle.reshard(True)
        elif fsdp_version(self.teacher_module) == 2:
            self.teacher_module.reshard()
        # Small samples from every shard are diagnostics only; no graph or loss changes.
        samples = [p.detach().flatten()[-64:].float().clone() for p in self.teacher_module.parameters()]
        super()._update_teacher()
        differences = [
            (p.detach().flatten()[-64:].float() - before).abs().sum().item()
            for p, before in zip(self.teacher_module.parameters(), samples, strict=True)
        ]
        count = sum(s.numel() for s in samples)
        self._teacher_sample_delta = sum(differences) / max(count, 1)
        if self.config.self_distillation.teacher_update_rate > 0:
            self._teacher_ema_updates += 1

    def update_policy(self, data):
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise RuntimeError("SDPO teacher was not initialized separately from the actor")
        before = self._teacher_ema_updates
        metrics = super().update_policy(data)
        grad_norms = metrics.get("actor/grad_norm", [])
        metrics.update({
            "self_distillation/upstream_implementation": 1.0,
            "self_distillation/verified_self_imitation_enabled": float(
                self.config.self_distillation.verified_self_imitation_coef > 0
            ),
            "self_distillation/teacher_is_separate": 1.0,
            "self_distillation/teacher_update_rate": self.config.self_distillation.teacher_update_rate,
            "self_distillation/teacher_ema_updates": self._teacher_ema_updates - before,
            "self_distillation/teacher_sample_mean_abs_delta": self._teacher_sample_delta,
            "self_distillation/optimizer_steps": len(grad_norms),
            "self_distillation/nonfinite_grad_steps": sum(not torch.isfinite(torch.tensor(v)) for v in grad_norms),
            "self_distillation/loss": metrics["actor/pg_loss"] / max(len(grad_norms), 1),
        })
        return metrics
