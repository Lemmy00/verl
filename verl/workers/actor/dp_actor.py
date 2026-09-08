# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import json
import logging
import os
import random
import time
import uuid
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _local_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Return the local tensor backing an ordinary or DTensor gradient."""
    return gradient.to_local() if isinstance(gradient, DTensor) else gradient


def _capture_gradient_probe(parameters, max_samples: int, offset: int = 0):
    """Sample accumulated gradients without copying the full model gradient.

    The returned samples represent the gradient from every objective accumulated before
    SFT replay.  Sampling the same coordinates after replay lets us recover the replay-only
    delta, including its norm and alignment with the other objectives.
    """
    if max_samples <= 0:
        return None
    gradients = []
    total_elements = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = _local_gradient(parameter.grad.detach()).reshape(-1)
        if gradient.numel() == 0:
            continue
        gradients.append((parameter, gradient.numel()))
        total_elements += gradient.numel()
    if total_elements == 0:
        return None

    # ceil(total / max_samples) keeps the probe at or below its configured memory budget.
    stride = max((total_elements + max_samples - 1) // max_samples, 1)
    sample_offset = int(offset) % stride
    cursor = 0
    entries = []
    for parameter, numel in gradients:
        first = (sample_offset - cursor) % stride
        if first < numel:
            gradient = _local_gradient(parameter.grad.detach()).reshape(-1)
            entries.append((parameter, first, gradient[first::stride].float().clone()))
        cursor += numel
    if not entries:
        return None
    return {"entries": entries, "stride": stride, "total_elements": total_elements}


@torch.no_grad()
def _finish_gradient_probe(probe, gradient_scale: float = 1.0) -> dict[str, float]:
    """Estimate non-replay, replay-only, and combined gradient geometry."""
    if probe is None:
        return {}
    entries = probe["entries"]
    device = entries[0][2].device
    # non-replay squared norm, replay squared norm, combined squared norm, dot product,
    # sampled coordinates, and represented coordinates.
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    stride = probe["stride"]
    for parameter, first, before in entries:
        if parameter.grad is None:
            combined = torch.zeros_like(before)
        else:
            combined = _local_gradient(parameter.grad.detach()).reshape(-1)[first::stride].float()
        if combined.numel() != before.numel():
            raise RuntimeError("Actor gradient layout changed during the SFT replay backward pass")
        replay = combined - before
        before64, replay64, combined64 = before.double(), replay.double(), combined.double()
        totals[0] += before64.square().sum()
        totals[1] += replay64.square().sum()
        totals[2] += combined64.square().sum()
        totals[3] += (before64 * replay64).sum()
        totals[4] += before.numel()
    totals[5] = probe["total_elements"]
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)

    sampled = float(totals[4].item())
    represented = float(totals[5].item())
    if sampled <= 0.0 or represented <= 0.0:
        return {}
    element_weight = represented / sampled
    scale = max(float(gradient_scale), 1.0)
    non_replay_sq = max(float(totals[0].item()) * element_weight, 0.0) / (scale * scale)
    replay_sq = max(float(totals[1].item()) * element_weight, 0.0) / (scale * scale)
    combined_sq = max(float(totals[2].item()) * element_weight, 0.0) / (scale * scale)
    dot = float(totals[3].item()) * element_weight / (scale * scale)
    non_replay_norm = non_replay_sq**0.5
    replay_norm = replay_sq**0.5
    combined_norm = combined_sq**0.5
    denom = non_replay_norm * replay_norm
    return {
        "actor/non_replay_grad_norm_est": non_replay_norm,
        "actor/sft_replay_grad_norm_est": replay_norm,
        "actor/combined_grad_norm_est": combined_norm,
        "actor/sft_replay_to_non_replay_grad_ratio": replay_norm / max(non_replay_norm, 1e-30),
        "actor/sft_replay_grad_cosine": dot / denom if denom > 0.0 else 0.0,
        "actor/sft_replay_grad_probe_fraction": sampled / represented,
    }


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        # SFT replay state. The pool is loaded lazily on first use so a disabled run pays
        # nothing, and the RNG is seeded PER RANK: identical seeds would have every rank
        # forward the same sequences, turning an N-sample step into one sample's gradient
        # replicated N times.
        self._sft_replay_pool: Optional[list] = None
        _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _sft_cfg = self.config.get("sft_replay", None)
        _sft_seed = int(getattr(_sft_cfg, "seed", 1234)) if _sft_cfg is not None else 1234
        self._sft_replay_seed = _sft_seed + _rank
        self._sft_replay_rng = random.Random(self._sft_replay_seed)

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns log-probs for response tokens, optionally with top-k/full log-probs
        needed by SDPO logit distillation.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("SDPO logit distillation requires disabling fused kernels.")
        model = module or self.actor_module

        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(getattr(model, "module", model).config, "vision_config")
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    inplace_backward = not calculate_entropy
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(int(distill_topk), logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                if calculate_sum_pi_squared:
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(int(distill_topk), logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _record_update(self, record: dict) -> None:
        """Append one JSON line per Adam update to $ACTOR_UPDATE_RECORD_PATH.rank{R}.

        The 2026-09-08 round-2 review showed the chart pipeline cannot preserve
        identifiable Adam-update records: reduce_metrics collapses the per-update
        lists to step means, and zero-eligible updates omit keys, so list indices are
        not update IDs. This recorder is the durable per-update channel: per-rank
        files (concurrent writers never share a file), append-mode, one line per
        optimizer step with explicit IDs. Disabled unless the env var is set.
        """
        path = self._resolve_update_record_path()
        if path is False:
            return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            # Diagnostics must never kill a training step; warn once per incident type.
            logger.warning("ACTOR_UPDATE_RECORD_PATH write failed: %s: %s", type(exc).__name__, exc)

    def _resolve_update_record_path(self):
        """Per-rank record path from ACTOR_UPDATE_RECORD_PATH, or False when disabled.

        Callers must gate record CONSTRUCTION on this: the record dict reads live
        optimizer state (param_groups), which stub optimizers in the actor test
        harnesses do not have -- and building a dict nobody writes is waste anyway.
        """
        path = getattr(self, "_update_record_path", None)
        if path is None:
            base = os.environ.get("ACTOR_UPDATE_RECORD_PATH", "").strip()
            if base:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                path = f"{base}.rank{rank}"
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                # Round-3 qualification: update_index/update_policy_call are actor-
                # local and restart from 1 after recreation. actor_session (fresh per
                # actor lifetime) plus the restored Adam step in each record make
                # restarts and replay-on/off forks distinguishable in the files.
                self._update_record_session = uuid.uuid4().hex[:12]
            else:
                path = False
            self._update_record_path = path
        return path

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    # ---------------------------------------------------------------------------------
    # SFT replay
    # ---------------------------------------------------------------------------------
    # Anchors a behaviour that RL erodes, by mixing supervised cross-entropy on real SFT
    # examples into the same optimizer step as the policy gradient.
    #
    # This is NOT feedback_loss. That term reweights tokens of the ROLLOUTS, so it can only
    # anchor how a rollout is written, not what the policy chooses to do -- and it worked
    # exactly that well: lean/feedback_quality/block_f1 held flat at 0.77-0.84 through a run
    # in which repair_rate collapsed 0.645 -> 0.242 and give_up_rate went 0.176 -> 0.535.
    # Replay forwards DIFFERENT sequences -- ones demonstrating "read the compiler error, try
    # again" -- which is why it needs its own forward pass rather than a weight vector.
    #
    # Gradients land before _optimizer_step(), so they accumulate into the same step and are
    # clipped by the same grad_clip. The pool is pre-tokenised by prepare_sft_replay.py: this
    # class has no tokenizer, and tokenising in the hot loop would cost more than the forward.
    def _load_sft_replay_pool(self, cfg) -> list:
        if self._sft_replay_pool is not None:
            return self._sft_replay_pool
        files = getattr(cfg, "files", None)
        if not files:
            self._sft_replay_pool = []
            return self._sft_replay_pool
        import pandas as pd

        # RAISE, do not degrade. Returning [] here would leave the run "enabled" in every
        # logged config while contributing nothing -- the exact silent failure the config
        # guard exists to prevent, and invisible for the ~25 steps it takes to notice that
        # actor/sft_replay_ce was never emitted. Better to die at the first optimizer step.
        try:
            df = pd.read_parquet(files)
        except Exception as exc:
            raise RuntimeError(
                f"sft_replay.enabled is set but the pool at {files!r} could not be read: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        n_repair = int(df["proof_repair"].sum()) if "proof_repair" in df.columns else -1

        def _row_dict(row):
            d = {"input_ids": list(row.input_ids), "prompt_len": int(row.prompt_len)}
            # Keep the class label with the row: the 2026-09-08 round-2 review caught
            # this loader dropping proof_repair, which silently turned every class-split
            # metric into "oneshot". Null/absent stays absent -> class "unknown", never
            # mislabeled one-shot.
            flag = getattr(row, "proof_repair", None)
            if flag is not None and not pd.isna(flag):
                d["proof_repair"] = bool(flag)
            return d

        self._sft_replay_pool = [
            _row_dict(row)
            for row in df.itertuples()
            # 1 <= prompt_len < len. The upper bound: a prompt covering the whole sequence
            # has no CE target and would contribute 0/0. The LOWER bound is subtler --
            # total_target_tokens counts len - prompt_len, but target = lbl[:, 1:] drops
            # position 0, so at prompt_len == 0 the denominator over-counts by one and the
            # row is silently underweighted. Pools from prepare_sft_replay.py always have
            # prompt_len >= 1 (measured min 99), so this only guards hand-made pools.
            if 1 <= int(row.prompt_len) < len(row.input_ids)
        ]
        if not self._sft_replay_pool:
            raise RuntimeError(
                f"sft_replay pool at {files!r} has {len(df)} rows but none usable "
                "(every row has prompt_len >= len(input_ids)); replay would be silently inert."
            )
        # is_initialized() guard matches __init__: get_rank() raises outright when there is
        # no process group, which makes the pool unloadable in any single-process context
        # (a smoke test, a notebook, a CPU debug run).
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # Report the repair share. It is the entire point of the pool, the builder can
            # backfill it away when one class runs short, and it is otherwise discarded here
            # and never visible anywhere downstream -- so a 12%-repair pool would train for
            # hours looking exactly like the 50% one that was asked for.
            share = f"{n_repair / max(len(df), 1):.1%}" if n_repair >= 0 else "unknown"
            print(
                f"SFT replay: loaded {len(self._sft_replay_pool)} sequences from {files} "
                f"(proof_repair {share})"
            )
        return self._sft_replay_pool

    def _sft_replay_step(self, pad_token_id: int) -> dict:
        cfg = self.config.get("sft_replay", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return {}
        # The reference policy shares this class and has no optimizer; it must never update.
        if self.actor_optimizer is None:
            return {}
        # The replay forward is the plain padded path: no Ulysses pad/slice, always 2-D
        # position_ids. Correct at sequence-parallel size 1 (this run), wrong and possibly
        # hanging above it, so refuse rather than train something subtly incorrect.
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError(
                "sft_replay does not support ulysses_sequence_parallel_size > 1: the replay "
                "forward does not shard sequences. Set sft_replay.enabled=false or size=1."
            )
        pool = self._load_sft_replay_pool(cfg)
        if not pool:
            return {}

        lam = float(getattr(cfg, "lambda_coef", 0.1))
        requested = int(getattr(cfg, "samples_per_step", 128))
        mb = max(int(getattr(cfg, "micro_batch_size", 1)), 1)
        if requested <= 0 or lam == 0.0:
            return {}
        # samples_per_step is GLOBAL. Each rank draws its share, so the replay:rollout ratio
        # is set by the config alone and does not scale 8x when the run moves from 1 GPU to
        # 8. world and requested are identical on every rank, so every rank computes the
        # same n and runs the same number of forward/backward calls -- no collective skew.
        world = (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )
        n = min(max(requested // world, 1), len(pool))

        device = get_device_id()
        # Draw from a stream keyed on the optimizer's own step counter rather than from the
        # long-lived RNG built in __init__. That RNG rewinds to draw zero on every resume, and
        # this pipeline resumes often (resume_mode=auto, SAVE_FREQ=25, both prior runs died
        # mid-flight) -- it would be the only sampling stream that rewinds, since verl already
        # checkpoints the RL dataloader. AdamW's "step" is restored with the optimizer state,
        # so it is monotonic across resumes with no new plumbing. Falls back to the sequential
        # RNG before the first optimizer step, when the state dict is still empty.
        step = None
        try:
            for st in self.actor_optimizer.state.values():
                t = st.get("step")
                if t is not None:
                    step = int(t.item() if hasattr(t, "item") else t)
                    break
        except (AttributeError, TypeError):
            step = None
        if step is None:
            rng = self._sft_replay_rng
        else:
            rng = random.Random((self._sft_replay_seed * 1_000_003) + step)
        picks = [pool[i] for i in rng.sample(range(len(pool)), n)]
        # Denominator for the WHOLE step, computed before the loop. Normalising per chunk
        # instead makes the loss a per-sequence mean at micro_batch_size=1 and a per-token
        # mean at full width -- different numbers whenever rows differ in length, which they
        # always do (pool median 1529 tokens, p90 4138). micro_batch_size is a memory knob
        # and must not move the gradient.
        total_target_tokens = sum(len(r["input_ids"]) - r["prompt_len"] for r in picks)
        if total_target_tokens <= 0:
            return {}
        # Update-wide token denominator across DP workers (2026-09-08 contract; before
        # this, each rank divided by ITS token count and FSDP's mean made the objective a
        # mean-of-per-rank-means, weighting ranks unequally when the v2 pool's long rows
        # land unevenly). Per-rank loss carries world * ce_sum / global_tokens so the
        # FSDP mean reduction yields the exact global token mean.
        if torch.distributed.is_initialized():
            replay_world = torch.distributed.get_world_size()
            _tok = torch.tensor([float(total_target_tokens)], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(_tok, op=torch.distributed.ReduceOp.SUM)
            global_target_tokens = float(_tok.item())
        else:
            replay_world = 1
            global_target_tokens = float(total_target_tokens)
        total_ce, total_tok = 0.0, 0
        class_ce = {"repair": 0.0, "oneshot": 0.0, "unknown": 0.0, "long": 0.0, "short": 0.0}
        class_tok = {"repair": 0, "oneshot": 0, "unknown": 0, "long": 0, "short": 0}

        for start in range(0, n, mb):
            chunk = picks[start : start + mb]
            width = max(len(r["input_ids"]) for r in chunk)
            ids = torch.full((len(chunk), width), pad_token_id, dtype=torch.long, device=device)
            att = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
            lbl = torch.full((len(chunk), width), -100, dtype=torch.long, device=device)
            for i, r in enumerate(chunk):
                seq, plen = r["input_ids"], r["prompt_len"]
                ids[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long, device=device)
                att[i, : len(seq)] = 1
                # Only the answer is a target. Training the model to reproduce the problem
                # statement would spend the replay budget on text it is already given.
                lbl[i, plen : len(seq)] = ids[i, plen : len(seq)]
            position_ids = (att.cumsum(dim=-1) - 1).clamp(min=0)

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                out = self.actor_module(
                    input_ids=ids,
                    attention_mask=att,
                    position_ids=position_ids,
                    use_cache=False,
                )
            logits = out.logits[:, :-1, :]
            target = lbl[:, 1:]
            valid = target != -100
            # NOT F.cross_entropy on .float() logits. At vocab 151,936 a worst-case chunk
            # (2 x 6144 tokens) costs 3.7 GB for the bf16 logits plus 7.5 GB for the float32
            # copy plus another 7.5 GB for log_softmax inside CE -- ~19 GB of transient peak
            # on a run that already sits at 70-122 GB of 143 GB. logprobs_from_logits is the
            # utility the main path uses: flash-attn CE with in-place backward where
            # available, row-wise chunking otherwise, and it never materialises a full
            # float32 softmax. Targets are clamped because -100 is not a gatherable index;
            # the mask, not the index, is what excludes them.
            logp = logprobs_from_logits(logits, target.masked_fill(~valid, 0))
            # torch.where, NOT `logp * valid`. Masked positions carry the log-prob of the
            # filler index 0, which at vocab 151,936 in bf16 can underflow to -inf -- and
            # -inf * False is NaN, not 0. One such position turns the whole step's loss into
            # NaN, and the gradient with it. Verified: (logp * valid).sum() -> nan where
            # torch.where(...).sum() -> the correct value.
            # sum(dtype=float32): with two 6144-token rows at initial CE ~log(151936)=11.9
            # the sum reaches ~1.5e5, past fp16's finite range -- and an inf here happens
            # BEFORE ShardedGradScaler.scale(), which a lower scale factor cannot repair.
            # bfloat16 (this run) is unaffected; the promotion is on the [B,T] logprob
            # tensor, not the vocab logits, so it costs nothing.
            ce_sum = -torch.where(valid, logp, torch.zeros_like(logp)).sum(dtype=torch.float32)
            n_tok = valid.sum()
            # Divided by the step's TOTAL target tokens, not this chunk's, so the chunks sum
            # to one token-weighted mean CE regardless of how they were split.
            loss = lam * ce_sum * replay_world / global_target_tokens
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_ce += ce_sum.detach().item()
            total_tok += int(n_tok.item())
            # Per-row CE for the class splits below (repair vs one-shot, long vs
            # short). The 2026-09-08 replay-gradient review showed the pool-mean CE
            # cannot see a 5.9-nat transition flip; these splits (and the offline
            # frozen-history probes) are the replacements -- diagnostics only, the
            # loss above is untouched.
            with torch.no_grad():
                row_ce = -torch.where(valid, logp, torch.zeros_like(logp)).sum(dim=-1)
                row_tok = valid.sum(dim=-1)
                for row_offset, pick in enumerate(chunk):
                    ce_v = float(row_ce[row_offset].item())
                    tok_v = int(row_tok[row_offset].item())
                    flag = pick.get("proof_repair")
                    key_class = "unknown" if flag is None else ("repair" if flag else "oneshot")
                    key_len = "long" if len(pick["input_ids"]) > 6144 else "short"
                    class_ce[key_class] += ce_v
                    class_tok[key_class] += tok_v
                    class_ce[key_len] += ce_v
                    class_tok[key_len] += tok_v

        if total_tok == 0:
            return {}
        _class_names = ("repair", "oneshot", "unknown", "long", "short")
        _sums = torch.tensor(
            [total_ce]
            + [class_ce[n] for n in _class_names]
            + [float(class_tok[n]) for n in _class_names],
            dtype=torch.float64,
            device=device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(_sums, op=torch.distributed.ReduceOp.SUM)
        global_ce_sum = float(_sums[0].item())
        class_ce_global = {n: float(_sums[1 + i].item()) for i, n in enumerate(_class_names)}
        class_tok_global = {n: float(_sums[6 + i].item()) for i, n in enumerate(_class_names)}
        mean_ce = total_ce / total_tok
        return {
            "actor/sft_replay_ce": mean_ce,
            "actor/sft_replay_loss": lam * mean_ce,
            "actor/sft_replay_tokens": float(total_tok),
            "actor/sft_replay_lambda": lam,
            "actor/sft_replay_tokens_global": float(global_target_tokens),
            # The OBJECTIVE's value: global token-mean CE, matching the loss's
            # denominator. actor/sft_replay_ce above stays the per-rank token mean
            # (a mean of these across ranks is NOT the objective when lengths differ).
            "actor/sft_replay_ce_global": global_ce_sum / max(global_target_tokens, 1.0),
            # Class splits: GLOBAL token means (class CE sums and token counts are
            # all-reduced above -- averaging per-rank means would be sensitive to which
            # workers draw the long rows). Globally empty classes omit their key; rows
            # with no proof_repair metadata land in "unknown", never in "oneshot". The
            # v2 pool halves the header token's share of the pool-mean (0.645% ->
            # 0.345% of answer tokens), so an aggregate CE is even less able to
            # surface transition damage than before -- read these, and the offline
            # header probes, instead of the aggregate.
            **{
                f"actor/sft_replay_ce_{name}": class_ce_global[name] / class_tok_global[name]
                for name in _class_names
                if class_tok_global[name] > 0
            },
        }

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(data.non_tensor_batch.get("multi_modal_inputs"))

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        # Per-update record IDs: (call, index-in-call) plus a monotone per-rank counter.
        self._update_policy_calls = getattr(self, "_update_policy_calls", 0) + 1
        _update_in_call = 0
        # The 2026-09-08 normalization contract is derived for seq-mean-token-mean,
        # where a microbatch aggregate is exactly the mean of per-row token-means over
        # eligible rows. Under token-mean or sum-normalized modes the same scaling is
        # NOT partition-invariant (verified by the phase-2 review); refuse rather than
        # silently produce packing-dependent gradients. Likewise a caller-supplied
        # global_batch_info would stack a second normalization on top of this one.
        if self.config.loss_agg_mode != "seq-mean-token-mean":
            raise NotImplementedError(
                "update_policy's update-wide denominators support only "
                f"loss_agg_mode='seq-mean-token-mean', got {self.config.loss_agg_mode!r}"
            )
        if self.config.get("global_batch_info", None):
            raise ValueError(
                "actor.global_batch_info must be empty: the 2026-09-08 update-wide "
                "denominators already normalize globally, and agg_loss-level global "
                "batch info would double-scale the objectives"
            )
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        self_distillation_enabled = loss_mode == "sdpo"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        if self_distillation_enabled:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            missing = self_distillation_required_keys - set(data.batch.keys())
            assert not missing, f"Missing required SDPO keys: {missing}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if "ppo_response_mask" in data.batch.keys():
            select_keys.append("ppo_response_mask")
        if "feedback_loss_weights" in data.batch.keys():
            select_keys.append("feedback_loss_weights")
        if "error_feedback_loss_weights" in data.batch.keys():
            select_keys.append("error_feedback_loss_weights")
        if "theorem_statement_loss_weights" in data.batch.keys():
            select_keys.append("theorem_statement_loss_weights")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(data.non_tensor_batch.get("multi_modal_inputs"))
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        # Objective metrics are appended once PER ADAM UPDATE as global means (the
        # numerators and denominators are all-reduced with matching weights); the
        # logged training step then averages its updates via reduce_metrics, which is
        # a labeled mean-over-updates, not a sum. The 2026-09-08 phase-2 review found
        # the previous scheme reported unweighted per-worker means (an empty worker
        # dragged the average toward zero) and summed the two Adam updates together.
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    # Normalise by the micro-batches this optimizer step actually holds.
                    # The config formula assumes every mini-batch is full, which stops
                    # being true as soon as the row count is not a multiple of
                    # ppo_mini_batch_size -- e.g. once feedback aux rows are appended --
                    # leaving the short tail step scaled by the wrong factor.
                    self.gradient_accumulation = max(len(micro_batches), 1)

                # ------------------------------------------------------------------
                # Explicit per-objective denominators for THIS Adam update (2026-09-08
                # normalization contract). The old scheme divided every objective by the
                # microbatch count, which counts aux/padding microbatches that contribute
                # nothing to a given objective: adding a zero-PG aux row in its own
                # microbatch HALVED a rollout row's PG gradient (reproduced on the real
                # backward), and PG/KL ran at ~0.51x effective strength for the entire
                # 2026-09 collapse series while the separately-normalized replay CE was
                # undiluted. Each objective now averages over ITS eligible rows across
                # the whole update, all DP workers included: per-rank losses carry
                # world_size * (local contribution) / N_global, so FSDP's mean gradient
                # reduction yields exactly sum_global / N_global. Repartitioning across
                # microbatches, appending inert rows, or uneven per-rank row counts can
                # no longer change any objective's gradient.
                # KL shares the PG row set by construction: kl_loss_mask.any(row) <=>
                # ppo_response_mask.any(row), since the ppo mask is response_mask with
                # zeros (subset), and the KL row gate keys on ppo_response_mask.any().
                _mb = mini_batch.batch
                _resp_mask = _mb["response_mask"]
                _ppo_mask = (
                    _mb["ppo_response_mask"] if "ppo_response_mask" in _mb.keys() else _resp_mask
                ).to(bool)
                _fb_w = _mb["feedback_loss_weights"] if "feedback_loss_weights" in _mb.keys() else None
                if bool((_ppo_mask & ~_resp_mask.to(bool)).any()):
                    raise AssertionError(
                        "ppo_response_mask must be a subset of response_mask; the KL "
                        "row-count sharing (and the KL row gate itself) rely on it"
                    )
                pg_rows_local = _ppo_mask.any(dim=-1).sum()
                fb_rows_local = (
                    (_fb_w.sum(dim=-1) > 0).sum() if _fb_w is not None else torch.zeros((), dtype=torch.long)
                )
                if torch.distributed.is_initialized():
                    dp_world_size = torch.distributed.get_world_size()
                    _counts = torch.tensor(
                        [int(pg_rows_local.item()), int(fb_rows_local.item())],
                        dtype=torch.float64,
                        device=get_device_id(),
                    )
                    torch.distributed.all_reduce(_counts, op=torch.distributed.ReduceOp.SUM)
                    pg_rows_global = float(_counts[0].item())
                    fb_rows_global = float(_counts[1].item())
                else:
                    dp_world_size = 1
                    pg_rows_global = float(pg_rows_local.item())
                    fb_rows_global = float(fb_rows_local.item())
                pg_rows_local_f = max(float(pg_rows_local.item()), 1.0)
                fb_rows_local_f = max(float(fb_rows_local.item()), 1.0)
                # Recorded every update: the coefficient conversation is only meaningful
                # against these. rollout_row_fraction is the old dilution factor, kept as
                # the audit trail of what this fix changed.
                append_to_dict(metrics, {
                    "actor/denom/pg_rows_global": pg_rows_global,
                    "actor/denom/feedback_rows_global": fb_rows_global,
                    "actor/denom/total_rows_global": float(_resp_mask.shape[0]) * dp_world_size,
                    "actor/denom/rollout_row_fraction": (
                        pg_rows_global / max(float(_resp_mask.shape[0]) * dp_world_size, 1.0)
                    ),
                })

                self.actor_optimizer.zero_grad()
                # Per-update objective numerators (local), reduced globally after the
                # microbatch loop with the SAME row weights the gradients use.
                pg_sum_local = 0.0
                kl_sum_local = 0.0
                kl_ppo_mask_sum_local = 0.0
                fb_sum_local = 0.0

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    ppo_response_mask = model_inputs.get("ppo_response_mask", response_mask).to(bool)
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = (
                        model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    )
                    if self_distillation_enabled:
                        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for SDPO distillation"


                    return_all_logps = (
                        self_distillation_enabled
                        and self_distillation_cfg.full_logit_distillation
                        and not self_distillation_cfg.distillation_topk
                    )
                    distill_topk = (
                        self_distillation_cfg.distillation_topk
                        if self_distillation_enabled and self_distillation_cfg.full_logit_distillation
                        else None
                    )

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=self.teacher_module or self.actor_module,
                            )
                        pg_loss, pg_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_outputs["log_probs"],
                            response_mask=ppo_response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_outputs.get("all_logps") if return_all_logps else None,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_outputs.get("topk_logps") if distill_topk else None,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                        )
                        # "Did this micro-batch distil anything?" -- and that is the
                        # COMPOSED mask, not the row mask alone. self_distillation_mask is
                        # a (bsz,) row gate that compute_self_distillation_loss multiplies
                        # into response_mask (= ppo_response_mask here, already narrowed by
                        # the generated-<feedback> spans, the last-closed-lean-block cut and
                        # the lean_valid_reward row zeroing). A row can therefore pass the
                        # row gate and still contribute zero tokens, in which case this used
                        # to report False while self_distillation/active_tokens read 0 --
                        # the two series contradicting each other on the same micro-batch.
                        # active_tokens is loss_mask.sum() over exactly the composed mask,
                        # so read the answer off it rather than recomputing a weaker one.
                        active_tokens = pg_metrics.get("self_distillation/active_tokens")
                        if active_tokens is not None:
                            pg_metrics["self_distillation/empty_target_batch"] = float(active_tokens) == 0.0
                        else:  # pragma: no cover - the loss always emits active_tokens
                            pg_metrics["self_distillation/empty_target_batch"] = (
                                float(self_distillation_mask.sum().detach().item()) == 0.0
                                if self_distillation_mask is not None
                                else False
                            )
                    elif ppo_response_mask.any():
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=ppo_response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                    else:
                        pg_loss = log_prob.sum() * 0.0
                        pg_metrics = {}
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        if ppo_response_mask.any():
                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=ppo_response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                    # Objective scaling per the update-wide contract above. pg_loss is
                    # agg_loss's mean over this microbatch's ELIGIBLE rows; x n_rows gives
                    # their sum, x world/N_global completes the global row-mean once FSDP
                    # averages gradients across ranks. Zero eligible rows anywhere keeps a
                    # finite zero through the dummy-loss branches.
                    pg_rows_micro = ppo_response_mask.any(dim=-1).sum().item()
                    pg_update_scale = (
                        float(pg_rows_micro) * dp_world_size / pg_rows_global if pg_rows_global > 0 else 0.0
                    )
                    policy_loss = pg_loss * pg_update_scale
                    if calculate_entropy and entropy is not None:
                        if ppo_response_mask.any():
                            entropy_agg = agg_loss(
                                loss_mat=entropy, loss_mask=ppo_response_mask, loss_agg_mode=loss_agg_mode
                            )
                        else:
                            entropy_agg = log_prob.sum() * 0.0
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        # The entropy collapse detector (0.019 -> 3.5) has to keep seeing
                        # the whole response: this value overrides the one computed over
                        # response_mask at old_log_prob time, and ppo_response_mask now
                        # ends at the last closed lean block.
                        with torch.no_grad():
                            if response_mask.any():
                                micro_batch_metrics["actor/entropy_full_response"] = agg_loss(
                                    loss_mat=entropy,
                                    loss_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                ).detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff * pg_update_scale

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        # The reference-KL brake aggregates over the FULL response_mask,
                        # NOT the narrowed ppo_response_mask. ppo_response_mask stops one
                        # token past the last closed ```lean4 block, so while it also gated
                        # the KL, every token in the masked tail was completely unanchored:
                        # no policy gradient AND no pull back toward the reference policy.
                        # That is where the 5.7e-03 -> 1.4e-01 drift lived. The KL is a
                        # divergence brake, not a credit-assignment signal -- it should hold
                        # every token the model actually emitted.
                        #
                        # The PPO policy-gradient term above (policy_loss_fn /
                        # compute_self_distillation_loss) and the entropy bonus deliberately
                        # KEEP ppo_response_mask. Crossing the two masks would put a policy
                        # gradient back on the tail, which is the entire bug the narrowing
                        # fixed. This KL term is the ONLY consumer of the full mask inside
                        # the loss.
                        #
                        # ROW GATE. The widening applies only to rows that carry a policy
                        # gradient at all. A row whose ppo_response_mask is ENTIRELY zero is
                        # not an ordinary rollout row with a narrowed tail -- it is one of
                        # the feedback / error-feedback / theorem-statement auxiliary CE
                        # rows appended by _maybe_append_feedback_aux_batch, and those rows
                        # carry NO reference log-prob. That function runs AFTER
                        # ref_log_prob is unioned into the batch and copies only
                        # prompts/responses/input_ids/attention_mask/position_ids/
                        # response_mask/teacher_*; every other key, ref_log_prob included,
                        # falls through to `torch.zeros_like`, so an aux row arrives with
                        # ref_log_prob == 0.0 exactly -- a placeholder, not the reference
                        # policy. Aggregating it into the KL is not a brake: it is a pull
                        # toward log p == 0, i.e. probability 1.0 on every auxiliary target
                        # token, which is unbounded memorisation pressure directly opposing
                        # their CE objective (and it hits the unweighted tokens too).
                        # Measured on the shipped config (low_var_kl, seq-mean-token-mean,
                        # kl_loss_coef 0.01): one aux row in a two-row micro-batch moves
                        # actor/kl_loss from 5.0e-05 to 2.19 and puts 636x more gradient on
                        # an aux token than on a rollout token. kld = 4.39 sits under the
                        # low_var_kl clamp of 10, so that gradient is live, not saturated.
                        # It would also drown actor/kl_loss as a drift detector -- the exact
                        # signal this change exists to restore.
                        #
                        # The gate is per-ROW and never per-token: a real rollout row keeps
                        # its ENTIRE response in the KL, masked tail included. That is the
                        # point of this change and the gate does not touch it. When no
                        # narrowing is in play, ppo_response_mask IS response_mask, every
                        # non-empty row passes the gate, and this is a no-op.
                        #
                        # Rows zeroed by valid_reward_mask also fail the gate and stay out
                        # of the KL. They do have a genuine ref_log_prob, so including them
                        # would be sound, but excluding them is exactly their pre-change
                        # treatment -- no regression, and it keeps the rule one sentence
                        # long: the KL holds the whole response of every row the policy
                        # gradient touches, and nothing else.
                        kl_row = ppo_response_mask.any(dim=-1, keepdim=True)
                        kl_loss_mask = response_mask.to(bool) & kl_row
                        if kl_loss_mask.any():
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=kl_loss_mask, loss_agg_mode=loss_agg_mode)
                        else:
                            kl_loss = log_prob.sum() * 0.0

                        # Same row set as PG (see the contract comment at the top of the
                        # minibatch): kl_loss is agg_loss's mean over rows with any
                        # kl_loss_mask token, scaled to the update-wide row mean.
                        kl_rows_micro = kl_loss_mask.any(dim=-1).sum().item()
                        kl_update_scale = (
                            float(kl_rows_micro) * dp_world_size / pg_rows_global if pg_rows_global > 0 else 0.0
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef * kl_update_scale
                        kl_sum_local += kl_loss.detach().item() * float(kl_rows_micro)
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        # actor/kl_loss is now itself the whole-response KL. Its old
                        # full-response continuity-copy key was retired in the 2026-09-08
                        # metric cleanup: phase 2 is a fresh baseline, and two names for
                        # one number read as two measurements.

                        # The genuinely new number: the same kld over the narrowed mask, i.e.
                        # the tokens that still receive a policy gradient. This is what
                        # actor/kl_loss used to report. Outside the graph; the gap between
                        # actor/kl_loss and this is the tail's contribution.
                        with torch.no_grad():
                            if ppo_response_mask.any():
                                ppo_mask_kl = agg_loss(
                                    loss_mat=kld,
                                    loss_mask=ppo_response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                kl_ppo_mask_sum_local += ppo_mask_kl.detach().item() * float(
                                    pg_rows_micro
                                )

                    feedback_cfg = self.config.get("feedback_loss", None)
                    feedback_enabled = (
                        bool(getattr(feedback_cfg, "enabled", False)) if feedback_cfg is not None else False
                    )
                    feedback_lambda = (
                        float(getattr(feedback_cfg, "lambda_coef", 1.0)) if feedback_cfg is not None else 1.0
                    )
                    feedback_weights = model_inputs.get("feedback_loss_weights", None)
                    if feedback_enabled and feedback_weights is not None:
                        feedback_weights = feedback_weights.to(device=log_prob.device, dtype=log_prob.dtype)
                        ce = -log_prob
                        feedback_weight_sum = feedback_weights.sum()
                        # Per-ROW weighted mean, then the update-wide mean over rows with
                        # targets (the 2026-09-08 contract). The previous microbatch-global
                        # sum(w*ce)/sum(w) changed value with row packing; per-row means are
                        # partition-invariant, and at micro_batch_size=1 (every run so far)
                        # they are numerically identical to the old behavior before scaling.
                        fb_row_wsum = feedback_weights.sum(dim=-1)
                        fb_row_mask = fb_row_wsum > 0
                        fb_rows_micro = int(fb_row_mask.sum().item())
                        if fb_rows_micro > 0 and fb_rows_global > 0:
                            fb_row_loss = (ce * feedback_weights).sum(dim=-1) / torch.clamp(fb_row_wsum, min=1e-8)
                            feedback_loss = fb_row_loss[fb_row_mask].mean()
                            fb_sum_local += fb_row_loss[fb_row_mask].sum().detach().item()
                            policy_loss = policy_loss + feedback_lambda * fb_row_loss[fb_row_mask].sum() * (
                                dp_world_size / fb_rows_global
                            )
                        else:
                            feedback_loss = (ce * feedback_weights).sum() * 0.0
                        micro_batch_metrics["actor/feedback_lambda"] = feedback_lambda
                        # Only report CE from micro-batches that actually carry feedback
                        # tokens. Averaging in the zeros from ordinary rollout rows made the
                        # logged CE under-read the true value by roughly 2x.
                        if feedback_weight_sum > 0:
                            micro_batch_metrics["actor/feedback_ce_loss"] = feedback_loss.detach().item()
                            micro_batch_metrics["actor/feedback_aux_loss"] = (
                                feedback_lambda * feedback_loss
                            ).detach().item()
                            # actor/feedback_active_tokens is NOT emitted. Reduced across
                            # the micro-batches that carry feedback it came out
                            # byte-identical to feedback/avg_feedback_tokens on all 226
                            # steps of the last run, and that name is the honest one: this
                            # one says "active tokens" but reports a per-micro-batch mean.
                            # actor/feedback_weight_sum stays -- it is the loss
                            # denominator, and it is NOT the same series (weights are
                            # real-valued, active_tokens counts the non-zero ones).
                            micro_batch_metrics["actor/feedback_weight_sum"] = feedback_weight_sum.detach().item()
                        error_weights = model_inputs.get("error_feedback_loss_weights", None)
                        if error_weights is not None:
                            error_weights = error_weights.to(device=log_prob.device, dtype=log_prob.dtype)
                            error_weight_sum = error_weights.sum()
                            error_loss = (ce * error_weights).sum() / torch.clamp(error_weight_sum, min=1.0)
                            if error_weight_sum > 0:
                                micro_batch_metrics["actor/error_feedback_ce_loss"] = error_loss.detach().item()
                                micro_batch_metrics["actor/error_feedback_active_tokens"] = (
                                    (error_weights > 0).sum().detach().item()
                                )
                                micro_batch_metrics["actor/error_feedback_weight_sum"] = (
                                    error_weight_sum.detach().item()
                                )
                        theorem_statement_weights = model_inputs.get("theorem_statement_loss_weights", None)
                        if theorem_statement_weights is not None:
                            theorem_statement_weights = theorem_statement_weights.to(
                                device=log_prob.device, dtype=log_prob.dtype
                            )
                            theorem_weight_sum = theorem_statement_weights.sum()
                            theorem_loss = (ce * theorem_statement_weights).sum() / torch.clamp(
                                theorem_weight_sum, min=1.0
                            )
                            if theorem_weight_sum > 0:
                                micro_batch_metrics["actor/theorem_statement_ce_loss"] = theorem_loss.detach().item()
                                micro_batch_metrics["actor/theorem_statement_active_tokens"] = (
                                    (theorem_statement_weights > 0).sum().detach().item()
                                )
                                micro_batch_metrics["actor/theorem_statement_weight_sum"] = (
                                    theorem_weight_sum.detach().item()
                                )

                    # No loss_scale_factor here any more: every objective already carries
                    # its explicit update-wide denominator (see the minibatch header). The
                    # old division by microbatch count is exactly the confirmed aux/padding
                    # dilution this fix removes, in both the static and dynamic-bsz branches.
                    loss = policy_loss
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    pg_sum_local += pg_loss.detach().item() * float(pg_rows_micro)
                    append_to_dict(metrics, micro_batch_metrics)

                # Probe the already-accumulated non-replay gradient on a small, fixed set of
                # coordinates. After replay backward, the change at those coordinates is the
                # replay-only gradient. This is an estimate, but unlike comparing scalar loss
                # values it directly measures gradient strength and alignment.
                sft_cfg = self.config.get("sft_replay", None)
                probe_samples = (
                    int(getattr(sft_cfg, "grad_probe_samples", 0))
                    if sft_cfg is not None and bool(getattr(sft_cfg, "enabled", False))
                    else 0
                )
                # Zero probe samples must be a true no-op: walking parameters() here
                # broke every stub-module test and costs a pointless traversal per
                # update when the probe is off.
                grad_probe = (
                    _capture_gradient_probe(
                        self.actor_module.parameters(), probe_samples, offset=self._sft_replay_seed
                    )
                    if probe_samples > 0
                    else None
                )

                # Before the optimizer step, so replay gradients accumulate into this same
                # step and are clipped by the same grad_clip rather than forming an
                # unclipped second update.
                # Per-update objective means, globally weighted (2026-09-08 P2 fix):
                # numerators reduced with the same eligible-row weights the gradients
                # carry, then divided by the ALREADY-reduced global counts. Appended
                # once per Adam update; reduce_metrics' mean over the logged step is a
                # labeled mean-over-updates. An empty worker contributes zero to the
                # numerator AND zero to the denominator instead of dragging a
                # per-worker average toward zero.
                if torch.distributed.is_initialized():
                    _sums = torch.tensor(
                        [pg_sum_local, kl_sum_local, fb_sum_local],
                        dtype=torch.float64,
                        device=get_device_id(),
                    )
                    torch.distributed.all_reduce(_sums, op=torch.distributed.ReduceOp.SUM)
                    pg_sum_global, kl_sum_global, fb_sum_global = (float(v) for v in _sums.tolist())
                else:
                    pg_sum_global, kl_sum_global, fb_sum_global = pg_sum_local, kl_sum_local, fb_sum_local
                update_metrics = {}
                if pg_rows_global > 0:
                    update_metrics["actor/pg_loss"] = pg_sum_global / pg_rows_global
                    update_metrics["actor/kl_loss"] = kl_sum_global / pg_rows_global
                    # Diagnostic, deliberately LOCAL (per-rank narrowed-mask KL mean).
                    update_metrics["actor/kl_loss_ppo_mask"] = (
                        kl_ppo_mask_sum_local / pg_rows_local_f
                    )
                if fb_rows_global > 0:
                    update_metrics["actor/feedback_ce_loss_update"] = fb_sum_global / fb_rows_global
                append_to_dict(metrics, update_metrics)

                sft_replay_metrics = self._sft_replay_step(pad_token_id)
                if sft_replay_metrics:
                    append_to_dict(metrics, sft_replay_metrics)
                if grad_probe is not None and sft_replay_metrics:
                    gradient_scale = self.scaler.get_scale() if self.scaler is not None else 1.0
                    append_to_dict(metrics, _finish_gradient_probe(grad_probe, gradient_scale))

                grad_norm = self._optimizer_step()
                grad_norm_val = float(grad_norm.detach().item())
                mini_batch_metrics = {"actor/grad_norm": grad_norm_val}
                append_to_dict(metrics, mini_batch_metrics)

                # Durable per-update record (see _record_update). Objective fields are
                # None when the update had no eligible rows -- explicit absence, never a
                # fabricated zero. grad_norm is the pre-clip total norm; clip_factor is
                # what scaled the applied gradient. Constructed ONLY when recording is
                # on: the dict reads optimizer state that stub harnesses do not stub.
                if self._resolve_update_record_path() is not False:
                    self._update_record_index = getattr(self, "_update_record_index", 0) + 1
                    _update_in_call += 1
                    _norm_finite = bool(torch.isfinite(grad_norm).item())
                    if self.scaler is not None:
                        _step_outcome = "scaler_managed"
                    elif _norm_finite:
                        _step_outcome = "applied"
                    else:
                        _step_outcome = "skipped_nonfinite"
                    _opt_step_val = None
                    try:
                        for _st in self.actor_optimizer.state.values():
                            _t = _st.get("step")
                            if _t is not None:
                                _opt_step_val = int(_t.item() if hasattr(_t, "item") else _t)
                                break
                    except (AttributeError, TypeError):
                        pass
                    self._record_update({
                        "ts": time.time(),
                        "actor_session": getattr(self, "_update_record_session", None),
                        "optimizer_step": _opt_step_val,
                        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                        "world": dp_world_size,
                        "global_steps": data.meta_info.get("global_steps", None),
                        "update_policy_call": self._update_policy_calls,
                        "update_in_call": _update_in_call,
                        "update_index": self._update_record_index,
                        "pg_rows_local": int(pg_rows_local.item()),
                        "feedback_rows_local": int(fb_rows_local.item()),
                        "total_rows_local": int(_resp_mask.shape[0]),
                        "pg_rows_global": pg_rows_global,
                        "feedback_rows_global": fb_rows_global,
                        "total_rows_global": float(_resp_mask.shape[0]) * dp_world_size,
                        "pg_loss": (pg_sum_global / pg_rows_global) if pg_rows_global > 0 else None,
                        "kl_loss": (kl_sum_global / pg_rows_global) if pg_rows_global > 0 else None,
                        "feedback_ce_loss": (fb_sum_global / fb_rows_global) if fb_rows_global > 0 else None,
                        "sft_replay_ce_global": sft_replay_metrics.get("actor/sft_replay_ce_global")
                        if sft_replay_metrics
                        else None,
                        "sft_replay_tokens_global": sft_replay_metrics.get("actor/sft_replay_tokens_global")
                        if sft_replay_metrics
                        else None,
                        "lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                        "grad_norm": grad_norm_val,
                        "grad_clip": float(self.config.grad_clip),
                        "clip_factor": (
                            min(1.0, float(self.config.grad_clip) / grad_norm_val)
                            if _norm_finite and grad_norm_val > 0
                            else None
                        ),
                        "step_outcome": _step_outcome,
                    })
        self.actor_optimizer.zero_grad()
        return metrics
