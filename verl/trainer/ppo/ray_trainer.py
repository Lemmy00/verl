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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import math
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _object_array_1d(values):
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    return arr


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _truthy(value) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _lean_valid_reward_mask(reward_extra_infos_dict: dict, length: int, device: torch.device) -> torch.Tensor | None:
    valid_flags = reward_extra_infos_dict.get("lean_valid_reward", None)
    if valid_flags is None or len(valid_flags) != length:
        return None
    return torch.tensor([_truthy(flag) for flag in valid_flags], dtype=torch.bool, device=device)


# The one status string that means "this rollout proved the requested theorem". It has a
# single producer in the reward. Deliberately NOT derived from the score: score == 1.0 is
# false for every verified rollout that paid any shaping penalty (the worst verified row
# scores 0.55), and score > 0 is a threshold over a ladder whose rungs move -- it is also
# true of renamed_declaration whenever RENAMED_DECLARATION_REWARD is nonzero (it has been
# 0.7; it is 0.0 today), and the shaping penalties push honest failures below zero.
_LEAN_VERIFIED_STATUS = "verified"

# ---------------------------------------------------------------------------------
# Metric-surface constants for _lean_reward_diagnostics.
#
# The last full run emitted 268 distinct step keys, of which 28 were zero on every one
# of its 226 steps and 28 groups were byte-identical series. The constants below are the
# de-duplication, and every one of them was re-derived from that run's worker log before
# anything was deleted -- not taken on trust.
# ---------------------------------------------------------------------------------

# Statuses whose rate is pinned to 0.0 rather than left absent. See the comment at the
# use site: each of these is the surviving name of a series that was dropped as a
# duplicate, so it has to be a continuous series and not a sparse one.
_LEAN_ALWAYS_REPORTED_STATUSES = ("verified", "lean_timeout")

# (reward_extra_info key, metric stem) for the per-rollout event counters.
#
# Each of these used to emit THREE keys -- _events, _per_rollout and _rollout_rate.
# _events is _per_rollout times the (constant, and separately logged) rollout count, and
# _per_rollout equals _rollout_rate exactly whenever no single rollout fires the counter
# more than once. So one key, lean/<stem>_rate, carries all of it in the common case.
#
# "candidate_timeout" is NOT in this table: lean/candidate_timeout_per_rollout and
# lean/candidate_timeout_rollout_rate were byte-identical to lean/status_rate/lean_timeout
# on all 226 steps, and lean/candidate_timeout_events was byte-identical to
# lean/status/lean_timeout -- a candidate timeout IS the lean_timeout status. Read those.
_LEAN_EVENT_COUNTERS = (
    ("lean_timeouts", "timeout"),
    ("lean_feedback_fallback_timeouts", "feedback_fallback_timeout"),
    ("lean_setup_timeouts", "setup_timeout"),
    ("lean_replay_timeouts", "replay_timeout"),
    ("lean_replay_failures", "replay_failure"),
    ("lean_retries", "retry"),
    ("lean_command_attempts", "command_attempt"),
)

# Counters where a single rollout demonstrably fires more than once, so the fraction of
# AFFECTED rollouts is strictly less information than the mean count and both are kept.
# Measured on the same 226 steps:
#   retry            -- step 64 logged 2 retry events across 1 affected rollout.
#   command_attempt  -- mean 1.10-1.26 attempts per rollout, i.e. the two series differ on
#                       226 of 226 steps.
# The other five never exceeded one event per rollout; they get _per_rollout only on a
# step where they actually do (see the emit site), so the information is never lost, it
# just does not cost a permanent series.
_LEAN_MULTI_EVENT_COUNTERS = frozenset({"retry", "command_attempt"})

# Infrastructure counters, as (metric key suffix, reward_extra_info key, reducer name).
# Zero on all 226 steps of the last run and not a property of the policy: they are the
# Lean executor's health, not the model's. Rolled into lean/infra_events_total, which is
# always emitted; the individual keys come back in full the moment that total moves.
_LEAN_INFRA_EVENT_SUMS = (
    ("replay_failure_events", "lean_replay_failures"),
    ("replay_timeout_events", "lean_replay_timeouts"),
    ("setup_timeout_events", "lean_setup_timeouts"),
)
_LEAN_INFRA_GAUGES = (
    ("warmup_attempts_total", "lean_warmup_attempts_total"),
    ("warmup_failures_total", "lean_warmup_failures_total"),
    ("restart_warmups_total", "lean_restart_warmups_total"),
    ("restart_warmup_failures_total", "lean_restart_warmup_failures_total"),
)
# The event counters above are also reported per-rollout by _LEAN_EVENT_COUNTERS; those
# rate keys are suppressed on a healthy step for the same reason.
_LEAN_INFRA_RATE_STEMS = ("replay_failure", "replay_timeout", "setup_timeout")

# The feedback-quality family, aggregated over the rows that were actually scored.
# Computed on EVERY training rollout and, until now, aggregated only into val-aux/ --
# visible once every 50 steps, on the validation set, and nowhere on the training
# distribution the policy is actually moving on.
_LEAN_FEEDBACK_QUALITY_METRICS = (
    "block_f1",
    "block_precision",
    "block_recall",
    "anchored_block_f1",
    "token_f1",
    "sequence_similarity",
    "gold_blocks",
    "predicted_blocks",
    "block_count_abs_error",
    "error_presence_correct",
)


def _lean_feedback_quality_metrics(reward_extra_infos_dict: dict) -> dict[str, float]:
    """Predicted-feedback quality on the TRAINING distribution, per step.

    These eleven scores are computed on every training rollout -- the per-row
    feedback_quality_* keys have been in reward_extra_info all along -- and were
    aggregated nowhere except val-aux/, i.e. once every 50 steps, on the validation
    set. The feedback objective is trained on every step; it was measurable on 2% of
    them, against a different distribution.

    Averaged over the SCORED rows only. Unscored rows carry 0.0 for every score (the
    key set has to be identical on every row or verl's per-key array build raises), so
    a plain batch mean is the true score multiplied by the scoring rate -- it moves
    when feedback quality moves and it moves just as far when the fraction of rows
    with a gold side to compare against moves, and nothing tells the two apart. That
    is what lean/feedback_quality/scored_rate is for, and it is the denominator every
    other key here is divided by.
    """
    metrics: dict[str, float] = {}
    scored_flags = reward_extra_infos_dict.get("feedback_quality_scored", None)
    if scored_flags is None or len(scored_flags) == 0:
        return metrics

    scored = [_truthy(flag) for flag in scored_flags]
    # Always emitted, including at 0.0: it is a statement about the DENOMINATOR, not
    # about feedback quality, so zero is the literal truth and never a misleading
    # score. It is also the only thing that explains why the keys below are absent.
    metrics["lean/feedback_quality/scored_rate"] = float(np.mean(scored))

    scored_rows = [index for index, flag in enumerate(scored) if flag]
    if not scored_rows:
        # No row had a gold side this step. Every score below would be an average over
        # an empty set, and emitting 0.0 would read as "the model predicted feedback
        # and got it entirely wrong" rather than "nothing was compared".
        return metrics

    for name in _LEAN_FEEDBACK_QUALITY_METRICS:
        values = reward_extra_infos_dict.get(f"feedback_quality_{name}", None)
        if values is None or len(values) != len(scored):
            continue
        selected = []
        for index in scored_rows:
            try:
                number = float(values[index])
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                selected.append(number)
        if selected:
            metrics[f"lean/feedback_quality/{name}"] = float(np.mean(selected))
    return metrics


def _lean_attempt_penalty_inputs(
    data: DataProto, length: int, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Per-row inputs for the all-fail-group attempt-penalty refund, or (None, None).

    Two columns, both already emitted unconditionally on EVERY row by the Lean reward, so
    this adds no key to reward_extra_info (a conditionally-present key kills the run --
    reward_extra_keys is read off row 0 and every row is then indexed by every key):

      * ``lean_excess_blocks_penalty`` -- the amount APPLIED to this row. Two neighbours
        in the same dict literal are wrong in ways nothing would report.
        ``lean_excess_blocks_penalty_coef`` is the knob, the same constant on every row,
        and a constant cancels exactly in ``score - group_mean``: advantages would come
        out bit-identical to today while the added-back metric reported a healthy 0.02.
        ``lean_penalty_total`` also carries the non-termination and give-up charges, so
        refunding it would cancel the give-up penalty inside precisely the all-fail
        groups that penalty was added to fix. The measured target for all-fail groups is
        corr(attempts, advantage) = -0.009 and NOT 0.000; that residual is the
        non-termination penalty, still correlating with attempts, and it is the
        arithmetic proof that only the excess term may be refunded.
      * ``lean_status`` -- the unshaped verdict; see _LEAN_VERIFIED_STATUS.

    Read out of ``data.non_tensor_batch`` at the point of USE rather than cached earlier
    in the step: rejection sampling can reindex the batch between the reward and the
    advantage, and a tensor parked in ``batch.batch`` would be silently misaligned.

    FAILS OPEN. Any missing, short or unparseable column returns (None, None) and
    disables the refund for this batch, leaving the advantage path exactly as it was.
    Failing the other way -- treating rows as not-verified -- would neutralise EVERY
    group in the batch, the loudest possible wrong answer. The caller reports a disabled
    batch as ``lean/attempt_penalty_neutralize_unavailable = 1.0``, so a permanently
    inert feature is visible rather than inferred from a flat curve.
    """
    non_tensor = getattr(data, "non_tensor_batch", None)
    if not non_tensor:
        return None, None
    penalties = non_tensor.get("lean_excess_blocks_penalty", None)
    statuses = non_tensor.get("lean_status", None)
    if penalties is None or statuses is None:
        return None, None
    if len(penalties) != length or len(statuses) != length:
        return None, None

    amounts: list[float] = []
    for raw in penalties:
        if isinstance(raw, np.generic):
            raw = raw.item()
        try:
            amount = float(raw)
        except (TypeError, ValueError):
            return None, None
        # A NaN refund would poison its whole group's mean and std, taking every row in
        # the group down with it; a negative one would charge a penalty never paid.
        if not math.isfinite(amount) or amount < 0.0:
            return None, None
        amounts.append(amount)

    flags: list[bool] = []
    for raw in statuses:
        if isinstance(raw, np.generic):
            raw = raw.item()
        if not isinstance(raw, str) or not raw:
            # A column that is not strings is not the status column. Reading it anyway
            # would mark every row not-verified and neutralise the whole batch.
            return None, None
        flags.append(raw.strip().lower() == _LEAN_VERIFIED_STATUS)

    return (
        torch.tensor(amounts, dtype=torch.float32, device=device),
        torch.tensor(flags, dtype=torch.bool, device=device),
    )


def _neutralize_attempt_penalty_enabled(config) -> bool:
    """Read algorithm.neutralize_attempt_penalty_in_all_fail_groups (declared default ON).

    Declared in the AlgoConfig dataclass and in BOTH trainer yamls so no leading ``+`` is
    needed at launch -- a key missing from the schema makes a plain Hydra override a
    struct error. With no algorithm config at all there is no declared default to honour,
    so the advantage path is left exactly as it was.
    """
    if config is None:
        return False
    try:
        raw = config.get("neutralize_attempt_penalty_in_all_fail_groups", True)
    except AttributeError:
        raw = getattr(config, "neutralize_attempt_penalty_in_all_fail_groups", True)
    return _truthy(raw)


_GENERATED_FEEDBACK_BLOCK_RES = [
    re.compile(r"--\s*<feedback>\n[\s\S]*?--\s*</feedback>\n?", re.MULTILINE),
    re.compile(r"/-\s*<feedback>\n[\s\S]*?</feedback>\s*-/\n?", re.MULTILINE),
    re.compile(r"<feedback>\n?[\s\S]*?</feedback>\s*", re.MULTILINE),
    re.compile(r"/-\s*<unsolved goals>\n[\s\S]*?</unsolved goals>\s*-/\n?", re.MULTILINE),
]


def _generated_feedback_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in _GENERATED_FEEDBACK_BLOCK_RES:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text or ""))
    if not spans:
        return []

    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _span_overlaps(start: int, end: int, spans) -> bool:
    return any(start < int(span_end) and end > int(span_start) for span_start, span_end in spans or [])


# Distinguishes "the byte table has not been built yet" from "it was built and is
# unusable" (None). A plain sentinel string would be compared against a numpy array.
_UNSET = object()

# How many inconclusive batches the byte table may be validated over before the
# truncation is given up on. Only batches in which NO row could be checked count.
_LEAN_TAIL_VALIDATION_ATTEMPTS = 8

# One-shot guards so the estimator warnings below are not repeated every step.
_LEAN_TAIL_ADV_WARNED = False
_LEAN_ATTEMPT_NEUTRALIZE_ADV_WARNED = False


def _as_int(value, default: int = -1) -> int:
    """Read an int out of a non_tensor_batch cell, which is an object array."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------------------
# Mask truncation after the last CLOSED ```lean4 block.
#
# The reward reads exactly ONE thing out of a rollout: the last CLOSED lean block
# (lean_code_utils._extract_last_lean_code). Nothing after that block's closing fence is
# ever compiled, scored, or even looked at. GRPO nevertheless broadcasts the single scalar
# advantage over EVERY generated token (compute_grpo_outcome_advantage:
# scores.unsqueeze(-1) * response_mask, and compute_response_mask returns the whole
# response), so those unread tokens carry gradient.
#
# The tail is NOT reinforced -- that reading is wrong and was measured to be wrong. A
# rollout that derails almost never verifies (6 of 2852 verified rollouts carried a tail;
# high-tail rows verified 0.000 at all 17 steps sampled), so the tail rides a NEGATIVE
# advantage, dA -0.08 to -0.63. The problem is the opposite: there is no RESTORING force.
# At step 165, 56 of 88 gibberish rows sat at advantage exactly 0.0, and 19 of 32 groups
# held a clean row and a garbage row at identical score -- the group has nothing to prefer.
# What little negative signal exists is then diluted by loss_agg_mode=seq-mean-token-mean,
# which spreads a derailed rollout's gradient 14-26x thinner per token (1.08e-5 vs 2.78e-4
# at step 165). So the degenerate mode neither pays nor gets corrected, and it plateaued at
# 20-43% of rollouts for 36 steps instead of decaying.
#
# Measured: training-rollout gibberish went 1.2% -> 46.9% in two steps (128 -> 130),
# entropy 0.019 -> 3.5, kl_loss 5.7e-03 -> 1.4e-01 -- but the trigger led the tail:
# actor/ppo_kl broke its steps-95..119 range at step 120 while entropy was still 0.0189.
# Optimizer divergence starts it; the unscored region is why it never recovers. Over steps
# 100-125 (6656 rollouts) rollouts that never stopped cleanly verified 0.1% of the time
# while burning a mean 9.29 closed blocks. Zeroing the mask here removes the dead zone and
# stops it diluting the gradient of the part that is actually scored.
#
# The reward manager owns the fence parser (the abandoned-draft rule is not expressible as
# a ``` count, and verl cannot import project code -- the manager is loaded by path into a
# Ray actor), so the boundary crosses the process boundary as data: lean_last_block_end_byte,
# a UTF-8 offset into the decoded response, positioned PAST the closing fence, with -1
# meaning "no closed block". The trainer owns only the byte -> token index conversion,
# plus the ONE token of grace it keeps past that index so that "what follows a closing
# fence" stays in the gradient at all -- see lean_tail_response_mask's third rule.
# --------------------------------------------------------------------------------------


def build_token_byte_lengths(tokenizer) -> Optional[np.ndarray]:
    """Bytes each token id contributes to `decode(ids, skip_special_tokens=True)`.

    A byte-level BPE tokenizer (Qwen3) stores every token as its bytes run through the
    GPT-2 byte->unicode map, which is one CHARACTER PER BYTE, so the surface length in the
    vocab IS the byte length. Building this table once turns the offset -> token index
    mapping into a cumsum + searchsorted over the ORIGINAL response ids: exact, cheap, and
    free of the assumption the feedback-span path makes (_proof_action_response_mask
    decodes, re-encodes, and trusts that offsets[i] lines up with responses[i]; it already
    counts its own violations in feedback/generated_feedback_tokenizer_mismatch_rows).

    Special ids contribute 0 so the table matches skip_special_tokens=True, which is what
    the reward manager decoded with. Returns None when the tokenizer will not hand over a
    vocab -- callers MUST then fail OPEN and truncate nothing.
    """
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    if not vocab:
        return None

    try:
        size = max(int(max(vocab.values())) + 1, int(getattr(tokenizer, "vocab_size", 0) or 0))
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None

    table = np.zeros(size, dtype=np.int32)
    for token, token_id in vocab.items():
        idx = int(token_id)
        if 0 <= idx < size:
            table[idx] = len(token)

    # Added tokens are stored as literal text rather than byte-mapped, so their surface
    # length is a CHARACTER count and can undercount bytes. Take their real encoding, and
    # zero the special ones because skip_special_tokens=True drops them from the string.
    for token_id, added in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
        idx = int(token_id)
        if not (0 <= idx < size):
            continue
        if getattr(added, "special", False):
            table[idx] = 0
        else:
            table[idx] = len(str(getattr(added, "content", added)).encode("utf-8"))
    for special_id in getattr(tokenizer, "all_special_ids", None) or []:
        idx = int(special_id)
        if 0 <= idx < size:
            table[idx] = 0

    return table


# The whitespace a rollout is allowed to put between its closing fence and its EOS,
# mirroring the reward's LEAN_MAX_TRAILING_WS (lean_code_utils.DEFAULT_MAX_TRAILING_WS).
# The trainer reads the SAME environment variable the reward manager reads rather than
# taking a second Hydra key, because two knobs for one rule is how they drift: a run that
# lowered the reward's allowance would otherwise keep force-keeping the EOS behind padding
# the reward had already stopped calling terminated.
_LEAN_MAX_TRAILING_WS_DEFAULT = 3

# ASCII whitespace bytes only. The reward's rule is unicode-aware (a trailing NBSP still
# counts as terminated), but the trainer sees BYTES and a byte table, and treating a
# multi-byte whitespace character as prose only ever makes this stricter: the EOS behind
# it loses its force-keep. Erring that way costs one stop-token gradient on a rollout that
# padded its proof with invisible characters; erring the other way is what this rule
# exists to stop.
_ASCII_WS_BYTES = frozenset(b" \t\n\r\x0b\x0c")


def _gpt2_byte_decoder() -> dict:
    """Inverse of the GPT-2 byte->unicode map: one surface CHARACTER back to one byte.

    Built here rather than imported so this file does not acquire a transformers
    internals dependency for fifteen deterministic lines. It is the same map
    build_token_byte_lengths already relies on -- that function trusts len(surface) to BE
    the byte length, which is only true because the map is one character per byte.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    mapped = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, mapped)}


def build_token_trailing_ws_bytes(tokenizer, token_byte_len) -> Optional[np.ndarray]:
    """Bytes of ASCII whitespace at the END of each token's contribution to the decode.

    Parallel to build_token_byte_lengths, sized identically, and read together with it:
    a token is ENTIRELY whitespace exactly when its trailing run covers its whole byte
    length. That makes every special id trivially whitespace (0 == 0), which is the right
    answer -- skip_special_tokens=True means they are not in the string the reward
    measured at all, so they cannot be the thing separating a fence from an EOS.

    Only the END of the token is measured because that is the only part that can be
    "after the fence": the boundary routinely lands INSIDE a token (Qwen3 merges the
    closing fence with the newlines behind it into one "```\\n\\n"), and what matters then
    is whether the part past the boundary is whitespace, not the whole token.

    Returns None when the tokenizer will not hand over a vocab, and the caller MUST then
    fail OPEN -- keeping the terminal EOS wherever it lands, exactly as before the
    adjacency rule. A missing table is not evidence of a garbage tail.
    """
    if token_byte_len is None:
        return None
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    if not vocab:
        return None

    table = np.asarray(token_byte_len)
    size = int(table.shape[0])
    if size <= 0:
        return None

    ws = np.zeros(size, dtype=np.int32)
    decoder = _gpt2_byte_decoder()
    for token, token_id in vocab.items():
        idx = int(token_id)
        if not (0 <= idx < size):
            continue
        run = 0
        for char in reversed(token):
            byte = decoder.get(char)
            if byte is None or byte not in _ASCII_WS_BYTES:
                break
            run += 1
        ws[idx] = run

    # Same two corrections build_token_byte_lengths makes, for the same reasons: added
    # tokens are stored as literal text rather than byte-mapped, and special ids
    # contribute nothing to the decoded string.
    for token_id, added in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
        idx = int(token_id)
        if not (0 <= idx < size):
            continue
        if getattr(added, "special", False):
            ws[idx] = 0
            continue
        run = 0
        for byte in reversed(str(getattr(added, "content", added)).encode("utf-8")):
            if byte not in _ASCII_WS_BYTES:
                break
            run += 1
        ws[idx] = run
    for special_id in getattr(tokenizer, "all_special_ids", None) or []:
        idx = int(special_id)
        if 0 <= idx < size:
            ws[idx] = 0

    return ws


def _tail_after_fence_is_permitted_whitespace(
    ids, cumulative_bytes, byte_table, ws_table, end_byte, boundary, eos_index, max_trailing_ws
) -> bool:
    """Is everything between the closing fence and the terminal EOS permitted whitespace?

    "Everything" is three pieces, and all three have to be checked or the answer is a
    guess: the part of the BOUNDARY token that sits past the fence (the fused "```\\n\\n"
    case), every whole token between it and the EOS (the one token of grace included --
    it is kept either way, but it is still part of the tail being judged), and the total
    length of that run.

    Bytes stand in for characters in the length bound. Every character is at least one
    byte, so a byte count can only OVER-estimate how many characters the tail holds, and
    over-estimating means refusing to force-keep -- the strict direction. For the ASCII
    whitespace this table recognises at all, the two counts are identical anyway.
    """
    total_bytes = int(cumulative_bytes[-1])
    trailing = total_bytes - end_byte
    if trailing < 0 or trailing > max_trailing_ws:
        # Checked FIRST because it is O(1) and it is what rejects the shape this rule
        # exists for: a 5000-token garbage tail fails here without touching the tokens.
        return False

    suffix = int(cumulative_bytes[boundary]) - end_byte
    if suffix > int(ws_table[ids[boundary]]):
        # The boundary token continues past the fence with something that is not
        # whitespace, e.g. "```" fused onto the first word of the tail.
        return False

    span = ids[boundary + 1 : eos_index]
    if span.size and not np.array_equal(ws_table[span], byte_table[span]):
        return False
    return True


def token_byte_lengths_agree_with_decode(
    tokenizer, table, responses, lengths, sample_rows: int = 8, scan_rows: int = 64
):
    """Check the byte table against real rows before trusting it for a whole run.

    A non-byte-level tokenizer (sentencepiece stores "\u2581" for a space, one char for
    one byte, but non-ASCII pieces are literal text) would produce a table that is subtly
    wrong, and a subtly wrong table cuts at the wrong token -- silently dropping real proof
    tokens from the gradient on a verified rollout. Cheaper to prove it once than to debug
    it never.

    Returns a TRI-STATE, and the caller must honour all three:
      True  -- the table reproduces real rows; trust it.
      False -- it demonstrably does not; disable the cut for the run.
      None  -- this batch carried no checkable row; ask again on a later batch rather
               than disabling the fix forever on one unlucky batch.

    Two per-row conditions are skipped rather than treated as evidence against the table,
    because a single such row used to disable the feature for an entire multi-day run:

    * an id outside the table. The table is sized from the tokenizer vocab (151669 on
      Qwen3) while the model config's vocab_size is larger (151936), so a sampled id can
      legitimately land above it. lean_tail_response_mask already skips such a row per row
      and counts it in fallback_rows.
    * a decode containing U+FFFD. Generation cut at max_response_length ends mid-character
      and decode then emits a 3-byte replacement where the table counts the 1 raw byte
      (measured: table 50 vs decode 52 on a one-token-short row). The table UNDERCOUNTS
      there, so the per-row cumsum lands LATE and the cut keeps extra tokens -- the safe
      direction -- while an exact byte comparison is simply meaningless.

    Both are most likely on exactly the collapsed batches this feature exists for, and on
    the first batch after a resume-from-collapse, which is the batch that decides the run.
    """
    checked = 0
    scanned = 0
    for row_idx, valid_len in enumerate(lengths):
        if checked >= sample_rows or scanned >= scan_rows:
            break
        valid_len = int(valid_len)
        if valid_len <= 0:
            continue
        scanned += 1
        ids = np.asarray(responses[row_idx, :valid_len]).astype(np.int64)
        if ids.size == 0 or int(ids.max()) >= table.shape[0] or int(ids.min()) < 0:
            continue
        try:
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        except Exception:
            continue
        if "\ufffd" in text:
            continue
        if int(table[ids].sum()) != len(text.encode("utf-8")):
            return False
        checked += 1
    return True if checked > 0 else None


def lean_tail_response_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    end_bytes,
    token_byte_len,
    eos_ids=(),
    response_chars=None,
    token_trailing_ws=None,
    max_trailing_ws: int = _LEAN_MAX_TRAILING_WS_DEFAULT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-token keep mask: 1 through ONE token past the last closed lean block, 0 after.

    Args:
        responses: (bs, response_length) generated token ids.
        response_mask: (bs, response_length) 1 for generated tokens, 0 for padding.
        end_bytes: per-row UTF-8 offset PAST the closing fence of the last closed block.
            -1 means NO CLOSED BLOCK and MUST be read as "keep the whole row".
        token_byte_len: table from build_token_byte_lengths.
        eos_ids: ids that end a generation; the terminal one is force-kept only where it
            is adjacent to what was kept (see below).
        response_chars: optional per-row character counts, used as a free drift guard.
        token_trailing_ws: table from build_token_trailing_ws_bytes, deciding whether the
            run between a closing fence and the EOS is whitespace. None fails OPEN: the
            terminal EOS is then force-kept wherever it lands, as it was before.
        max_trailing_ws: how much of that whitespace is permitted, in bytes. Mirrors the
            reward's LEAN_MAX_TRAILING_WS, which is the allowance a rollout can pad its
            proof with and still be scored TERMINATED.

    Returns (mask, stats). The mask is bool and is meant to be composed MULTIPLICATIVELY
    with response_mask / ppo_response_mask, never assigned over them.

    Three rules are load-bearing:

    * -1 keeps the FULL row. Zeroing a row that has no closed block would make it vanish
      from the sequence average (agg_loss drops rows with seq_mask == 0), so the -0.10
      no_lean_code penalty would produce no gradient at all and the exact degenerate mode
      this change exists to punish would become unpunishable.
    * The terminal EOS is force-kept ONLY where it is ADJACENT to what was kept, or
      separated from the closing fence by nothing but permitted whitespace (at most
      max_trailing_ws bytes of it -- the same allowance the reward's termination rule
      gives, read from the same knob). skip_special_tokens=True hides the EOS from the
      decoded string, so it always sorts after the boundary and a naive cut removes it --
      deleting all gradient on "stop here", which is the one behaviour the non-termination
      penalty is trying to teach.

      An UNCONDITIONAL carve-out, though, is worse than the disease on the shape it was
      not written for. "Closing fence + 5000 garbage tokens + EOS" leaves exactly two
      islands of gradient in the row, the proof and that EOS, and the EOS collects the
      ROW's scalar advantage. A derailed row sits BELOW its group mean (it pays the
      non-termination penalty, and rollouts that never stop cleanly verify 0.1% of the
      time), so that advantage is NEGATIVE and the force-keep spends it teaching the
      policy NOT to emit its stop token -- inside the same loss whose stated purpose is
      to teach it to stop. Masking the distant EOS costs nothing the row was owed: it
      still pays the penalty through its score, it simply stops carrying a gradient
      against stopping.

      Failing open here means KEEPING the EOS. With no whitespace table (a tokenizer
      that will not hand over a usable vocab) the trainer cannot tell padding from prose,
      and deleting a real stop token is the more expensive mistake. Rows whose EOS is
      dropped are counted in eos_masked_rows.
    * ONE token past the boundary token is kept, always. That token is the model's answer
      to "what comes after a closing fence", and it is the only token in the row that can
      carry that gradient: on a clean rollout it IS the EOS (reinforced on a verified
      proof, rather than merely spared by the force-keep above), and on a derailed one it
      is the first token of the garbage tail (pushed DOWN whenever the row sits below its
      group mean, which is where the non-termination penalty puts it). Cutting at the
      boundary instead leaves the closing fence as the last token with a gradient, and
      then nothing in the loss ever says whether to stop -- the non-termination penalty
      would be pricing a behaviour the mask had already made unlearnable.

      KNOWN AND ACCEPTED EXPOSURE: a rollout that VERIFIES but does not terminate scores
      1.0 - 0.05 = 0.95, keeps a positive advantage, and therefore has its first GARBAGE
      token reinforced.

      Frequency, with the denominator stated because the two available measurements do
      not share one. "Not terminated" USED to mean "ran to the generation cap", and those
      rows verify 0.1% of the time (steps 100-125, 6656 rollouts). Under the strict
      trigger it also covers rows that stopped cleanly on EOS but wrote a tail past their
      proof, and those are drawn from the stop-cleanly population, which verifies 44.6%
      of the time -- a completely different distribution, so the 0.1% is a floor and not
      the answer. The repo's own direct count of the new shape is the closer one: 6 of
      2852 verified rollouts carried a tail, 0.21% (lean_code_utils.last_closed_lean_block_end).
      Expect roughly 2-3x the old figure, and expect lean/verified_not_terminated_rate to
      read HIGHER than 0.1% on the first step purely from the definition change.

      Accepted deliberately at that size as the cost of making termination learnable at
      all. lean/verified_not_terminated_rate is the series to watch, and its denominator
      is the whole batch, not non-terminated rows.

      A second population sits inside the same rule and is NOT covered by that metric:
      the claim above that the garbage token is "pushed down" assumes the row is below
      its group mean, and a flat 0.05 does not guarantee that. Rows with no closed block
      are never truncated (end_byte == -1) but still enter the group at -0.15, so in an
      all-failure group a closed-block row at -0.05 sits ABOVE the mean and has its first
      tail token reinforced too. That is larger than the verified slice and only the
      verified slice is measured.

    Every failure path fails OPEN (no truncation) and is counted, because a boundary that
    lands EARLY silently starves real proof tokens on a verified rollout.
    """
    mask = torch.ones_like(response_mask, dtype=torch.bool)
    stats = {
        "rows": 0.0,
        "truncated_rows": 0.0,
        "no_block_rows": 0.0,
        "fallback_rows": 0.0,
        "eos_masked_rows": 0.0,
        "masked_tokens": 0.0,
        "response_tokens": 0.0,
    }
    if token_byte_len is None or end_bytes is None:
        return mask, stats

    table = np.asarray(token_byte_len)
    vocab_size = int(table.shape[0])
    eos = {int(token_id) for token_id in (eos_ids or ())}

    # A table of the wrong size cannot be indexed by the same ids, so treat it as absent
    # rather than half-trusting it: absent means the EOS carve-out behaves as it did
    # before the adjacency rule, which is the failing-open direction.
    ws_table = None if token_trailing_ws is None else np.asarray(token_trailing_ws)
    if ws_table is not None and int(ws_table.shape[0]) != vocab_size:
        ws_table = None
    try:
        max_trailing_ws = max(int(max_trailing_ws), 0)
    except (TypeError, ValueError):
        max_trailing_ws = _LEAN_MAX_TRAILING_WS_DEFAULT

    responses_np = responses.detach().cpu().numpy()
    lengths = response_mask.detach().sum(dim=1).long().cpu().numpy()
    batch_size = int(responses_np.shape[0])

    for row_idx in range(batch_size):
        valid_len = int(lengths[row_idx])
        stats["response_tokens"] += float(valid_len)
        if valid_len <= 0:
            continue
        stats["rows"] += 1.0

        end_byte = _as_int(end_bytes[row_idx], -1)
        if end_byte < 0:
            stats["no_block_rows"] += 1.0
            continue

        ids = responses_np[row_idx, :valid_len].astype(np.int64)
        if int(ids.max()) >= vocab_size or int(ids.min()) < 0:
            stats["fallback_rows"] += 1.0
            continue

        cumulative_bytes = np.cumsum(table[ids].astype(np.int64))
        total_bytes = int(cumulative_bytes[-1])
        if total_bytes <= 0 or end_byte > total_bytes:
            # The boundary cannot exceed the row it was measured on. If it does, the two
            # sides disagree about what this row says -- do nothing.
            stats["fallback_rows"] += 1.0
            continue
        if response_chars is not None:
            chars = _as_int(response_chars[row_idx], -1)
            if 0 <= total_bytes < chars:
                # UTF-8 is never fewer bytes than characters, so this can only mean the
                # byte table is wrong for this vocabulary. This direction is the SAFE one
                # -- an under-reporting table makes the cumulative sum reach end_byte
                # LATE, so the boundary lands late and extra tokens are kept -- and it is
                # caught anyway because a guard that only fires on the harmless half of a
                # symmetric impossibility is not a guard.
                stats["fallback_rows"] += 1.0
                continue
            if chars >= 0 and total_bytes > 4 * chars:
                # The other half, and the half that can silently eat a proof: UTF-8 is at
                # most 4 bytes per character, so an OVER-reporting table is equally
                # impossible. Its failure mode is the dangerous one -- the cumulative sum
                # reaches end_byte EARLY, the boundary is placed early, and the END OF A
                # VALID PROOF is masked off with nothing firing. Same treatment: fail
                # open, count it, and let mask/map_fallback_rows say the two sides
                # disagree about what this row says.
                stats["fallback_rows"] += 1.0
                continue

        # cumulative_bytes[i] is the byte count THROUGH token i, so the first index whose
        # cumulative count reaches end_byte is the token holding the boundary. Keep it
        # whole: a boundary landing mid-token keeps one extra token rather than cutting a
        # real one.
        keep = int(np.searchsorted(cumulative_bytes, end_byte, side="left")) + 1

        # keep is a COUNT of kept tokens, so keep + 1 is the first index dropped: the
        # boundary token, plus the one token after it (see the third rule above), survive.
        cut = keep + 1
        if cut >= valid_len:
            # Nothing sits past the extra kept token. Note this is the ordinary clean
            # shape "``` <EOS>", not an edge case: it must leave the row untouched AND
            # uncounted, or mask/truncated_row_rate reads ~1.0 on a perfect batch.
            continue

        mask[row_idx, cut:] = False
        masked = valid_len - cut
        last_idx = valid_len - 1
        if int(responses_np[row_idx, last_idx]) in eos:
            # ADJACENCY, not "wherever it landed" -- see the second rule above. An EOS
            # behind a garbage tail is the only token of that tail still carrying a
            # gradient, and on a below-mean row that gradient trains the policy not to
            # stop. last_idx <= cut is the EOS sitting immediately after the kept region;
            # the whitespace path is the row that padded its proof and stopped, which the
            # reward still scores as TERMINATED and which must keep its stop gradient.
            if (
                last_idx <= cut
                or ws_table is None
                or _tail_after_fence_is_permitted_whitespace(
                    ids,
                    cumulative_bytes,
                    table,
                    ws_table,
                    end_byte,
                    keep - 1,
                    last_idx,
                    max_trailing_ws,
                )
            ):
                mask[row_idx, last_idx] = True
                masked -= 1
            else:
                stats["eos_masked_rows"] += 1.0
        if masked <= 0:
            # The only token past the extra kept one was the terminal EOS, which is
            # force-kept: NOTHING was removed from this row. Counting it would make
            # mask/truncated_row_rate read ~1.0 on a nearly clean batch -- a well-formed
            # rollout ends "```\n<EOS>" and the EOS carries 0 bytes under
            # skip_special_tokens, so it always sorts past the boundary -- killing the one
            # metric that has to answer "how many rollouts carried a garbage tail".
            continue
        stats["truncated_rows"] += 1.0
        stats["masked_tokens"] += float(masked)

    return mask, stats


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # Only the GRPO branch below consumes lean_tail_mask. Swapping the estimator would
    # otherwise disable the truncation silently, with every mask/* metric still reporting
    # a cut that no longer reaches any advantage.
    global _LEAN_TAIL_ADV_WARNED
    if (
        not _LEAN_TAIL_ADV_WARNED
        and data.batch.get("lean_tail_mask", None) is not None
        and adv_estimator != AdvantageEstimator.GRPO
    ):
        _LEAN_TAIL_ADV_WARNED = True
        logger.warning(
            "adv_estimator=%s does not consume lean_tail_mask; the mask truncation after the "
            "last closed lean block does NOT reach the advantage in this branch.",
            adv_estimator,
        )
    # Same failure mode for the all-fail-group refund: only the GRPO branch below passes
    # it through, and grpo_vectorized takes neither valid_reward_mask nor these arguments.
    # Without this the feature would silently stop applying on an estimator swap while
    # every lean/attempt_penalty_* series kept reporting from a previous run's shape.
    global _LEAN_ATTEMPT_NEUTRALIZE_ADV_WARNED
    if (
        not _LEAN_ATTEMPT_NEUTRALIZE_ADV_WARNED
        and adv_estimator != AdvantageEstimator.GRPO
        and _neutralize_attempt_penalty_enabled(config)
        and "lean_excess_blocks_penalty" in getattr(data, "non_tensor_batch", {})
    ):
        _LEAN_ATTEMPT_NEUTRALIZE_ADV_WARNED = True
        logger.warning(
            "adv_estimator=%s does not consume the Lean attempt-penalty refund; "
            "algorithm.neutralize_attempt_penalty_in_all_fail_groups has NO effect in this branch.",
            adv_estimator,
        )

    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Do not hand the scalar advantage to tokens the reward never read. Everything
        # after the last CLOSED ```lean4 block is invisible to the reward, so on a verified
        # rollout it would otherwise collect the same +advantage as the proof itself.
        # Composed multiplicatively and only over the tail: unlike ppo_response_mask this
        # deliberately leaves the generated-feedback spans and the invalid-reward rows
        # alone, since those are separate decisions with their own metrics.
        # NOTE: the scalar per-row score is read as token_level_rewards.sum(-1) BEFORE any
        # mask is applied (compute_grpo_outcome_advantage), and the reward is written at
        # index valid_response_length-1 -- inside the truncated tail. Masking here changes
        # only the broadcast, never the score. Do not start masking token_level_rewards.
        lean_tail_mask = data.batch.get("lean_tail_mask", None)
        if lean_tail_mask is not None:
            grpo_calculation_mask = grpo_calculation_mask * lean_tail_mask.to(grpo_calculation_mask.dtype)

        # (D) Refund the excess-attempt penalty inside groups that solved nothing. The
        # two per-row columns are read here, the last point before use, and the group
        # decision itself is made inside compute_grpo_outcome_advantage so that it shares
        # one definition of group membership with the loop that builds the groups -- see
        # the block above id2score there. Both columns are None when the flag is off, and
        # then the estimator takes its previous code path exactly.
        attempt_penalty = None
        verified = None
        neutralize_stats: dict[str, float] | None = None
        if _neutralize_attempt_penalty_enabled(config):
            token_level_rewards = data.batch["token_level_rewards"]
            attempt_penalty, verified = _lean_attempt_penalty_inputs(
                data, token_level_rewards.shape[0], token_level_rewards.device
            )
            neutralize_stats = {}
            if (attempt_penalty is None or verified is None) and any(
                key.startswith("lean_") for key in getattr(data, "non_tensor_batch", {})
            ):
                # The refund is disabled for this batch. Say so as a metric rather than
                # leaving it to be guessed from a flat neutralised-group rate. Gated on
                # the batch carrying SOME lean_ column, so an upstream non-Lean run does
                # not log a lean/ metric on every step just because the flag defaults on;
                # any regression that drops one of the two columns still trips it, because
                # lean_valid_reward and the rest of the reward's extra info remain.
                neutralize_stats["lean/attempt_penalty_neutralize_unavailable"] = 1.0

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            valid_reward_mask=data.batch.get("valid_reward_mask", None),
            # Must be forwarded: without it grpo_adv_std_floor silently stays 0.0 and
            # the group-std floor never activates. The same hazard applies to the
            # attempt-penalty flag, which is read off this very object.
            config=config,
            attempt_penalty=attempt_penalty,
            verified=verified,
            neutralize_stats=neutralize_stats,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if neutralize_stats:
            # Handed back through meta_info because compute_advantage returns a DataProto,
            # not a metrics dict. The caller pops it in the same 'adv' timer block, so the
            # metric lands on the step it describes.
            data.meta_info["lean_attempt_neutralize_stats"] = neutralize_stats
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        sdpo_cfg = config.actor_rollout_ref.actor.get("self_distillation", {})
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = sdpo_cfg.get("reprompt_truncation", "right")

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        def json_safe(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {str(k): json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(v) for v in value]
            return value

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: json_safe(v[i]) for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _lean_reward_diagnostics(self, reward_tensor: torch.Tensor, reward_extra_infos_dict: dict) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if not reward_extra_infos_dict:
            return metrics

        def numeric_values(key: str) -> list[float]:
            values = []
            for value in reward_extra_infos_dict.get(key, []):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    values.append(number)
            return values

        statuses = list(reward_extra_infos_dict.get("lean_status", []))
        if statuses:
            total = max(len(statuses), 1)
            counts = {s: statuses.count(s) for s in set(statuses)}
            # The status families are sparse on purpose -- a status nothing produced this
            # step gets no key, so a panel appearing IS the event. Two of them are the
            # exception: they are now the ONLY carrier of a series that used to have a
            # second, always-present name (lean/valid_proof_rate and
            # lean/candidate_timeout_*, both dropped as exact duplicates), and a run that
            # verifies nothing or times out on nothing is precisely the step you must not
            # lose the point on. Pinned to 0.0 so those two series never gap.
            for status in _LEAN_ALWAYS_REPORTED_STATUSES:
                counts.setdefault(status, 0)
            for status, count in defaultdict(int, counts).items():
                metrics[f"lean/status/{status}"] = count
                metrics[f"lean/status_rate/{status}"] = count / total
            # lean/valid_proof_rate was byte-identical to lean/status_rate/verified on all
            # 226 steps of the last run, by construction (same numerator, same
            # denominator). Dropped; read lean/status_rate/verified.
            metrics["lean/infra_failure_rate"] = sum("infra" in str(status) for status in statuses) / total

        # Self-repair rate: the fraction of rollouts that completed a SECOND attempt.
        # Logged live because it is the earliest and sharpest collapse signal we have. In
        # qwen3-lean-feedback-grpo-v2 it rose 0.4% -> 68.4% by step 120, then fell to 0.4% by
        # step 150 -- and at step 150 it was 58% of VERIFIED proofs that had used 2+ attempts,
        # so losing it is a capability loss. actor/entropy and actor/kl_loss only moved later
        # and less sharply, which is why the collapse was not caught until step 200.
        closed_blocks = numeric_values("lean_closed_blocks")
        if closed_blocks:
            n = len(closed_blocks)
            metrics["lean/repair_rate"] = sum(1 for c in closed_blocks if c >= 2) / n
            metrics["lean/attempts_mean"] = sum(closed_blocks) / n
            metrics["lean/no_closed_block_rate"] = sum(1 for c in closed_blocks if c < 1) / n

        # ATTEMPTS STARTED, closed or abandoned -- the count the excess-block ramp is
        # priced on, which lean/attempts_mean above is NOT. The gap between them is the
        # whole reason this metric exists: dropping the intermediate closing fences turns
        # eight attempts into one closed block with an identical scored body, an
        # identical Lean verdict and an identical mask boundary. That dodge is now priced
        # (lean_code_utils.lean_block_attempt_count), but if it ever appears anyway it is
        # invisible everywhere else -- lean/attempts_mean FALLS, lean/repair_rate FALLS
        # and lean/penalized_rate FALLS, so the batch reads as a policy that learned to
        # be concise. lean/unclosed_block_rate is the only series that rises.
        #
        # It also silently disables the tail mask on the same rows: no closed block means
        # lean_last_block_end_byte == -1, which is the fail-open path, so watch this
        # against mask/no_closed_block_rows.
        attempt_blocks = numeric_values("lean_block_attempts")
        if attempt_blocks:
            n = len(attempt_blocks)
            metrics["lean/attempt_blocks_mean"] = sum(attempt_blocks) / n
            if len(closed_blocks) == n:
                unclosed = [a - c for a, c in zip(attempt_blocks, closed_blocks, strict=True)]
                metrics["lean/unclosed_blocks_mean"] = sum(unclosed) / n
                metrics["lean/unclosed_block_rate"] = sum(1 for u in unclosed if u > 0) / n

        # In-fence comment volume, in non-whitespace characters of the last closed block.
        #
        # The shaping this sits beside charges a tail AFTER the closing fence 0.05 and
        # masks it out of the gradient. The identical text as a /- ... -/ comment INSIDE
        # the block costs nothing: it is stripped before judging (LEAN_STRIP_COMMENTS),
        # so it does not move the Lean verdict, the block count or termination, and it
        # sits before the mask boundary so on a verified row it keeps the full +2.83
        # advantage. Nothing prices it. As the tail migrates inward, every metric added
        # for the termination and mask work moves the RIGHT way -- non_termination_rate
        # and truncated_token_rate both FALL -- so this series is the only contradiction
        # available. Read it against lean/block_body_chars_mean; the policy already emits
        # predicted /- <feedback> -/ blocks on 10.5% of rows, so it is not starting from
        # zero and the level means nothing. The SLOPE is the signal.
        # READ THE OTHER SERIES, NOT THIS ONE, for the migration above. Measured over
        # steps 110/116/120 of qwen3-sft-feedback-grpo-lr2e6, this total is 16,079,033
        # in-fence comment characters of which 15,839,001 -- 98.5% -- are the
        # /- <feedback> -/ blocks the SFT format requires and the feedback arm trains
        # deliberately. It is 98.5% normal behaviour, so a full doubling of genuine
        # rambling moves it by 1.5% and nothing here would show it. Kept as the
        # denominator and for continuity of the existing series, not as the detector.
        comment_chars = numeric_values("lean_block_comment_chars")
        if comment_chars:
            metrics["lean/block_comment_chars_mean"] = float(np.mean(comment_chars))
        # The same count with the /- <feedback> -/ annotations subtracted. Its SLOPE is
        # readable, which the total's is not. This is the series that moves if the tail
        # migrates inside the fence.
        #
        # IT DOES NOT START AT ZERO -- do not read a non-zero level as a dodge appearing.
        # Measured over 11,503 dumped rollouts from four runs: 3.51% of in-fence comment
        # characters are non-feedback, 52.5% of rows non-zero, per row mean 159.7 / p50
        # 28 / p90 339 / p99 1,900. That is ordinary proof narration, which the SFT data
        # contains too. A live ~160 is the baseline; watch the slope against it.
        #
        # The two are not redundant and neither replaces the other. Feedback volume
        # rising (the objective working) moves the total and leaves this flat; rambling
        # moving inward moves this and is lost in the total's noise. No comment COUNT is
        # priced -- charging for comments would tax the required output format. Comment
        # CONTENT is not equally insulated: a ```lean4 fence inside an in-fence comment
        # counts as an attempt for lean/attempt_blocks_mean and is priced by the ramp.
        # That is the fence scanner's behaviour, not this metric's; the gap between
        # lean/attempts_mean and lean/attempt_blocks_mean is where it shows.
        other_comment_chars = numeric_values("lean_block_other_comment_chars")
        if other_comment_chars:
            metrics["lean/block_other_comment_chars_mean"] = float(
                np.mean(other_comment_chars)
            )
        body_chars = numeric_values("lean_block_body_chars")
        if body_chars:
            metrics["lean/block_body_chars_mean"] = float(np.mean(body_chars))

        # Reward SHAPING, separated from capability. lean/reward_mean is the PENALIZED
        # score, so on the step this landed it steps down by up to 0.15 with no change in
        # solving ability; without lean/base_reward_mean beside it a shaping-driven level
        # shift and a real capability move are indistinguishable, and there is only one
        # run in which to tell them apart.
        base_scores = numeric_values("lean_base_score")
        if base_scores:
            metrics["lean/base_reward_mean"] = float(np.mean(base_scores))

        penalty_totals = numeric_values("lean_penalty_total")
        if penalty_totals:
            metrics["lean/penalty_total_mean"] = float(np.mean(penalty_totals))
            metrics["lean/penalized_rate"] = float(np.mean([p > 0 for p in penalty_totals]))

        # lean_terminated carries the STRICT signal: the row's last non-padding token is a
        # configured stop id AND the decoded text past the last closed block is at most
        # LEAN_MAX_TRAILING_WS (3) whitespace characters -- BOUNDED, because an unbounded
        # allowance let a policy pad after its proof for free. The reward manager emits
        # the flag under both this name and
        # lean_terminated_strict with the same value; this metric is read off the name
        # verl has always used so the W&B series does not break at the changeover.
        terminated_flags = reward_extra_infos_dict.get("lean_terminated", None)
        terminated = None
        if terminated_flags is not None and len(terminated_flags) > 0:
            terminated = [_truthy(v) for v in terminated_flags]
            terminated_rate = float(np.mean(terminated))
            metrics["lean/terminated_rate"] = terminated_rate
            # The rate the non-termination penalty is actually charged at.
            metrics["lean/non_termination_rate"] = 1.0 - terminated_rate

            # The exposure the mask's "keep one token past the closing fence" rule takes
            # on knowingly: these rows verify, so they keep a POSITIVE advantage, and the
            # token that rule keeps is the first token of a garbage tail rather than an
            # EOS -- so it is REINFORCED.
            #
            # DO NOT compare this against the 0.1% the design notes quote. That figure is
            # per rollout that RAN TO THE CAP, and it was measured before the strict
            # trigger widened "not terminated" to include rows that stopped cleanly with a
            # tail -- which come from the population that verifies 44.6% of the time, not
            # 0.1%. The comparable direct count is 0.21% (6 of 2852 verified rollouts
            # carried a tail), and this metric's denominator is the whole batch on top of
            # that, so multiply by lean/non_termination_rate before comparing with either.
            # Logged so a move off it is visible rather than inferred; the level at the
            # changeover is a definition change, the slope after it is the signal.
            if statuses and len(statuses) == len(terminated):
                verified_and_open = [
                    str(status) == "verified" and not flag for status, flag in zip(statuses, terminated, strict=True)
                ]
                metrics["lean/verified_not_terminated_rate"] = float(np.mean(verified_and_open))

        # Condition 1 of the strict trigger on its own: the row ended on a stop id,
        # whatever it wrote before it. The GAP between this and lean/terminated_rate is
        # the population the strict trigger newly charges -- rollouts that stopped, but
        # not at their proof -- and it is the number to read on the first step after a
        # termination-rule change, because a drop in terminated_rate alone cannot say
        # whether the model stopped stopping or merely stopped stopping CLEANLY.
        stopped_flags = reward_extra_infos_dict.get("lean_stopped_on_eos", None)
        if stopped_flags is not None and len(stopped_flags) > 0:
            stopped = [_truthy(v) for v in stopped_flags]
            metrics["lean/stopped_on_eos_rate"] = float(np.mean(stopped))
            if terminated is not None and len(terminated) == len(stopped):
                stopped_elsewhere = [flag and not done for flag, done in zip(stopped, terminated, strict=True)]
                metrics["lean/stopped_not_at_block_rate"] = float(np.mean(stopped_elsewhere))

        # The LOOSEST of the three termination signals and a pure drift metric: a policy
        # can satisfy it by appending three backticks. Kept beside the other two precisely
        # so the divergence is visible.
        eval_stop_flags = reward_extra_infos_dict.get("ends_with_eval_stop", None)
        if eval_stop_flags is not None and len(eval_stop_flags) > 0:
            metrics["lean/ends_with_eval_stop_rate"] = float(
                np.mean([_truthy(v) for v in eval_stop_flags])
            )

        no_termination_charges = numeric_values("lean_no_termination_penalty")
        if no_termination_charges:
            metrics["lean/no_termination_penalty_mean"] = float(np.mean(no_termination_charges))

        excess_charges = numeric_values("lean_excess_blocks_penalty")
        if excess_charges:
            metrics["lean/excess_block_penalty_mean"] = float(np.mean(excess_charges))
            metrics["lean/excess_block_rate"] = float(np.mean([c > 0 for c in excess_charges]))

        # The give-up charge: failed, exactly one closed block, no trailing unclosed
        # opener. Its own rate series rather than only its contribution to
        # lean/penalty_total_mean, because after the all-fail neutralisation below
        # (neutralize_attempt_penalty_in_all_fail_groups) the excess ramp is REFUNDED
        # inside all-fail groups -- 44.1% of groups at the time of measurement -- which
        # leaves this charge as the only per-row shaping term still separating rows
        # there. It is the single term the (D)+(G) pair is betting on, so it is the last
        # one that should be invisible. Folded into penalty_total it is
        # indistinguishable from non-termination: at the measured 16.4% incidence it is
        # 0.164 * 0.05 = 0.0082 of mean reward, well under the non-termination term's
        # contribution, so nothing separates "firing as designed" from "never fires".
        give_up_charges = numeric_values("lean_give_up_penalty")
        if give_up_charges:
            metrics["lean/give_up_penalty_mean"] = float(np.mean(give_up_charges))
            metrics["lean/give_up_rate"] = float(np.mean([c > 0 for c in give_up_charges]))

        error_kinds = [
            str(value)
            for value in reward_extra_infos_dict.get("lean_error_kind", [])
            if str(value)
        ]
        if error_kinds:
            total = max(len(statuses), len(error_kinds), 1)
            for kind in set(error_kinds):
                count = error_kinds.count(kind)
                metrics[f"lean/error_kind/{kind}"] = count
                # lean/error_kind_rate/wall_timeout was byte-identical to
                # lean/timeout_rollout_rate on all 226 steps of the last run -- same
                # population, counted twice -- so the RATE is dropped for that one kind
                # and lean/timeout_rate carries it. The COUNT stays, because the rest of
                # this family's counts stay and a hole in lean/error_kind/* would read as
                # "wall timeouts stopped happening".
                if kind != "wall_timeout":
                    metrics[f"lean/error_kind_rate/{kind}"] = count / total

        valid_flags = reward_extra_infos_dict.get("lean_valid_reward", None)
        if valid_flags is not None and len(valid_flags) > 0:
            metrics["lean/valid_reward_rate"] = float(np.mean([_truthy(v) for v in valid_flags]))

        canonical_flags = reward_extra_infos_dict.get("has_canonical_feedback", None)
        if canonical_flags is not None and len(canonical_flags) > 0:
            metrics["feedback/canonical_success_rate"] = float(np.mean([_truthy(v) for v in canonical_flags]))

        # ONE key per counter: lean/<stem>_rate, the fraction of rollouts that fired it.
        # _events and _per_rollout are recoverable from it whenever no rollout fires the
        # counter twice, which is the case for five of the seven; the two where it is not
        # keep _per_rollout beside the rate, and any other counter that starts
        # multi-firing gets it back on the step it does. Nothing is lost: a healthy step
        # logs 6 of these (three of the eight stems are infrastructure and gated below,
        # and candidate_timeout is gone entirely) where it used to log 24.
        infra_rate_keys: dict[str, float] = {}
        for key, metric_name in _LEAN_EVENT_COUNTERS:
            values = numeric_values(key)
            if not values:
                continue
            rate = float(np.mean([value > 0 for value in values]))
            multi_fired = any(value > 1 for value in values)
            if metric_name in _LEAN_INFRA_RATE_STEMS:
                # Held back and emitted below only if the infra roll-up is non-zero.
                infra_rate_keys[f"lean/{metric_name}_rate"] = rate
                if metric_name in _LEAN_MULTI_EVENT_COUNTERS or multi_fired:
                    infra_rate_keys[f"lean/{metric_name}_per_rollout"] = float(np.mean(values))
                continue
            metrics[f"lean/{metric_name}_rate"] = rate
            if metric_name in _LEAN_MULTI_EVENT_COUNTERS or multi_fired:
                # The rate counts AFFECTED ROLLOUTS, so on a step where one rollout fired
                # twice it is strictly less than the mean count and the two series stop
                # meaning the same thing. That is the only condition under which the
                # second key earns its place.
                metrics[f"lean/{metric_name}_per_rollout"] = float(np.mean(values))

        for key, metric_name in (
            ("lean_context_s", "context_wait_s"),
            ("lean_verify_s", "verify_wall_s"),
            ("lean_total_s", "total_wall_s"),
            ("reward_remote_s", "reward_remote_s"),
        ):
            values = numeric_values(key)
            if values:
                metrics[f"lean/{metric_name}/mean"] = float(np.mean(values))
                metrics[f"lean/{metric_name}/p50"] = float(np.percentile(values, 50))
                metrics[f"lean/{metric_name}/p90"] = float(np.percentile(values, 90))
                metrics[f"lean/{metric_name}/max"] = float(max(values))

        for key, metric_name in (
            ("lean_cache_hit", "problem_cache_hit_rate"),
            ("lean_context_cache_hit", "context_cache_or_env_hit_rate"),
        ):
            values = reward_extra_infos_dict.get(key, None)
            if values is not None and len(values) > 0:
                metrics[f"lean/{metric_name}"] = float(
                    np.mean([_truthy(value) for value in values])
                )

        executor_workers = numeric_values("lean_executor_workers")
        if executor_workers:
            metrics["lean/executor_workers"] = float(max(executor_workers))

        # ---- infrastructure roll-up -------------------------------------------------
        #
        # Replay failures, replay timeouts, setup timeouts and the four warmup gauges are
        # the Lean executor's health, not the policy's behaviour, and they were 0.0 on all
        # 226 steps of the last run. Thirteen permanently flat panels train the reader to
        # stop looking, which is the worst possible state for a metric whose entire job is
        # to fire once.
        #
        # So: ONE always-present gauge. It is a raw event count, not a rate, so it is
        # never a fraction of a batch that happened to be small, and it is emitted even
        # when every underlying key is missing (0.0) so the series never gaps. The instant
        # it moves off zero, every individual key comes back in full on that same step --
        # nothing is deleted, it is gated.
        infra_detail: dict[str, float] = dict(infra_rate_keys)
        infra_total = 0.0
        for metric_name, key in _LEAN_INFRA_EVENT_SUMS:
            values = numeric_values(key)
            if values:
                total_events = float(sum(values))
                infra_detail[f"lean/{metric_name}"] = total_events
                infra_total += total_events
        for metric_name, key in _LEAN_INFRA_GAUGES:
            values = numeric_values(key)
            if values:
                # A cumulative process-level gauge echoed on every row: max, not sum.
                gauge = float(max(values))
                infra_detail[f"lean/{metric_name}"] = gauge
                infra_total += gauge
        metrics["lean/infra_events_total"] = infra_total
        if infra_total > 0:
            metrics.update(infra_detail)

        # lean/loss_reward_mean was byte-identical to critic/rewards/mean AND to
        # critic/score/mean on all 226 steps -- all three are reward_tensor.sum(-1).mean().
        # The critic pair is upstream verl's and stays; this third name is dropped. The
        # VALUE is still computed here because lean/reward_mean falls back to it when the
        # reward manager emits no lean_score, and in that configuration it is the only
        # reward series this fork has.
        loss_reward_mean = reward_tensor.sum(dim=-1).float().mean().detach().item()
        lean_scores = reward_extra_infos_dict.get("lean_score", None)
        if lean_scores is not None and len(lean_scores) > 0:
            metrics["lean/reward_mean"] = float(np.mean([float(score) for score in lean_scores]))
        else:
            metrics["lean/reward_mean"] = loss_reward_mean

        metrics.update(_lean_feedback_quality_metrics(reward_extra_infos_dict))
        return metrics

    @staticmethod
    def _collect_solutions_by_uid(
        batch: DataProto,
        reward_tensor: torch.Tensor,
        success_reward_threshold: float,
        valid_sample_flags: Optional[list[bool]] = None,
    ) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            if valid_sample_flags is not None and idx < len(valid_sample_flags) and not valid_sample_flags[idx]:
                continue
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL)

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int,
        use_fallback_environment_feedback: bool = True,
        valid_reward_flags: Optional[list[bool]] = None,
    ) -> list[str | None]:
        feedback_list: list[str | None] = [None] * batch_size
        if not include_environment_feedback or not reward_extra_infos_dict:
            return feedback_list

        canonical = reward_extra_infos_dict.get("canonical_annotated_code", [])
        has_canonical = reward_extra_infos_dict.get("has_canonical_feedback", [])
        statuses = reward_extra_infos_dict.get("lean_status", [])
        clean_code = reward_extra_infos_dict.get("clean_lean_code", [])
        for idx in range(batch_size):
            if valid_reward_flags is not None and idx < len(valid_reward_flags) and not valid_reward_flags[idx]:
                continue
            if idx < len(canonical) and idx < len(has_canonical) and _truthy(has_canonical[idx]):
                text = canonical[idx]
                if isinstance(text, str) and text.strip():
                    feedback_list[idx] = text
                    continue
            if not use_fallback_environment_feedback:
                continue
            if idx < len(statuses):
                status = str(statuses[idx])
                if status and status != "verified":
                    code = clean_code[idx] if idx < len(clean_code) and isinstance(clean_code[idx], str) else ""
                    feedback = f"Lean verifier status: {status}."
                    if code.strip():
                        feedback += f"\n\nAttempted Lean code:\n```lean4\n{code.strip()}\n```"
                    feedback_list[idx] = feedback
        return feedback_list

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids,
        response_texts: list[str],
        dont_reprompt_on_self_success: bool,
        remove_thinking_from_demonstration: bool,
    ) -> str | None:
        solution_idxs = list(success_by_uid[uids[idx]])
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if not solution_idxs:
            return None
        solution = response_texts[solution_idxs[0]]
        if remove_thinking_from_demonstration:
            solution = self._remove_thinking_trace(solution)
        return solution

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode != "sdpo":
            return None

        device = batch.batch["input_ids"].device
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        batch_size = len(batch)

        response_texts = []
        response_lengths: list[int] = []
        for ids, mask in zip(responses.detach().cpu(), response_mask.detach().cpu(), strict=True):
            valid_len = int(mask.sum().item())
            response_lengths.append(valid_len)
            response_texts.append(self.tokenizer.decode(ids[:valid_len], skip_special_tokens=True))

        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
        extra_infos = batch.non_tensor_batch.get("extra_info", _object_array_1d([{} for _ in range(batch_size)]))
        prompt_texts: list[str] = []
        for idx in range(batch_size):
            if raw_prompts is not None and len(raw_prompts[idx]) > 0:
                prompt_texts.append(raw_prompts[idx][-1]["content"])
            else:
                extra = extra_infos[idx] if isinstance(extra_infos[idx], dict) else {}
                prompt_texts.append(str(extra.get("question", "")))

        raw_valid_reward_flags = reward_extra_infos_dict.get("lean_valid_reward") if reward_extra_infos_dict else None
        if raw_valid_reward_flags is not None and len(raw_valid_reward_flags) == batch_size:
            valid_reward_flags = [_truthy(flag) for flag in raw_valid_reward_flags]
        else:
            valid_reward_flags = [True] * batch_size

        def optional_int(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
                return None
            return int(value)

        skip_clipped_responses = _truthy(self_distillation_cfg.get("skip_clipped_responses", False))
        max_response_len = int(responses.shape[1])
        max_target_response_len = optional_int(self_distillation_cfg.get("max_target_response_len", None))
        # A rollout is clipped when it hits *whichever* ceiling binds first. vLLM
        # caps generation at max_model_len - prompt_len (vllm_async_server.py), so
        # when prompt_length + response_length exceeds max_model_len a long prompt
        # runs out of context before reaching max_response_len. Comparing against
        # max_response_len alone would then silently admit a context-truncated
        # response as a valid self-distillation target.
        rollout_cfg = self.config.actor_rollout_ref.get("rollout", None)
        max_model_len = optional_int(rollout_cfg.get("max_model_len", None)) if rollout_cfg else None
        prompt_width = int(batch.batch["input_ids"].shape[1]) - max_response_len
        if max_model_len is not None and prompt_width > 0:
            prompt_lengths = batch.batch["attention_mask"][:, :prompt_width].sum(dim=1).tolist()
            response_caps = [
                min(max_response_len, max(0, max_model_len - int(prompt_lengths[idx]))) for idx in range(batch_size)
            ]
        else:
            response_caps = [max_response_len] * batch_size
        clipped_response_flags = [
            skip_clipped_responses and response_lengths[idx] >= response_caps[idx] for idx in range(batch_size)
        ]
        too_long_target_flags = [
            max_target_response_len is not None and response_lengths[idx] > max_target_response_len
            for idx in range(batch_size)
        ]
        target_valid_flags = [
            valid_reward_flags[idx] and not clipped_response_flags[idx] and not too_long_target_flags[idx]
            for idx in range(batch_size)
        ]

        feedback_list = self._collect_feedback(
            include_environment_feedback=_truthy(self_distillation_cfg.get("include_environment_feedback", True)),
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
            use_fallback_environment_feedback=_truthy(
                self_distillation_cfg.get("use_fallback_environment_feedback", True)
            ),
            valid_reward_flags=target_valid_flags,
        )
        success_by_uid = self._collect_solutions_by_uid(
            batch,
            reward_tensor,
            success_reward_threshold=float(self_distillation_cfg.get("success_reward_threshold", 1.0)),
            valid_sample_flags=target_valid_flags,
        )
        solution_strs = [
            (
                self._get_solution(
                    idx,
                    success_by_uid,
                    batch.non_tensor_batch["uid"],
                    response_texts,
                    _truthy(self_distillation_cfg.get("dont_reprompt_on_self_success", False)),
                    _truthy(self_distillation_cfg.get("remove_thinking_from_demonstration", False)),
                )
                if target_valid_flags[idx]
                else None
            )
            for idx in range(batch_size)
        ]

        feedback_only_without_solution = _truthy(
            self_distillation_cfg.get("environment_feedback_only_without_solution", True)
        )

        def build_teacher_messages(idx: int) -> list[dict[str, str]]:
            system_messages = []
            if raw_prompts is not None:
                system_messages = list(raw_prompts[idx][:-1])

            has_solution = solution_strs[idx] is not None
            has_feedback = feedback_list[idx] is not None
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)
            feedback_format = self_distillation_cfg.get("environment_feedback_format", "generic")

            if use_feedback and feedback_format == "sft_proof_repair":
                reprompt_text = self_distillation_cfg.proof_repair_template.format(
                    prompt=prompt_texts[idx].rstrip(),
                    failed_attempt=feedback_list[idx].strip(),
                )
                return system_messages + [{"role": "user", "content": reprompt_text}]

            solution_section = ""
            if has_solution:
                solution_section = self_distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[idx]
                )

            feedback_section = ""
            if use_feedback:
                feedback_section = self_distillation_cfg.feedback_template.format(feedback_raw=feedback_list[idx])

            if has_solution or use_feedback:
                reprompt_text = self_distillation_cfg.reprompt_template.format(
                    prompt=prompt_texts[idx],
                    solution=solution_section,
                    feedback=feedback_section,
                )
            else:
                reprompt_text = prompt_texts[idx]

            return system_messages + [{"role": "user", "content": reprompt_text}]

        messages = [build_teacher_messages(idx) for idx in range(batch_size)]
        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})

        # Budget the teacher sequence as a WHOLE (reprompt + response) rather than
        # carving a fixed slice for the prompt. Two reasons:
        #  * correctness -- the reprompt ends with the trailing Lean error, the
        #    closing fence and the generation marker, and _build_annotated_lean
        #    emits its single error block last, so right-truncation destroys 100%
        #    of the repair signal it exists to convey.
        #  * memory -- the teacher forward materialises [T, vocab] logits, so the
        #    peak is driven by reprompt + response, not by either alone.
        # Over-budget reprompts are elided in the MIDDLE, keeping the head (question,
        # opening fence, imports, theorem statement) and the tail (trailing error,
        # closing fence, instruction, generation marker).
        max_reprompt_len = int(self_distillation_cfg.get("max_reprompt_len", 4096))
        max_teacher_total_len = optional_int(self_distillation_cfg.get("max_teacher_total_len", None))
        min_reprompt_len = int(self_distillation_cfg.get("min_reprompt_len", 1024))

        def _tokenize_reprompt(msgs: list[dict[str, str]]) -> list[int]:
            try:
                ids = self.tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True, **apply_kwargs
                )
            except TypeError:
                # Retrying without apply_kwargs silently changes what the teacher is
                # conditioned on (e.g. drops a chat_template override), so say so rather
                # than degrading quietly.
                logger.warning(
                    "apply_chat_template rejected data.apply_chat_template_kwargs=%s; "
                    "retrying WITHOUT them -- the teacher reprompt will use the tokenizer's "
                    "own chat template, which may not match the policy's prompt format.",
                    sorted(apply_kwargs),
                )
                ids = self.tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            return list(ids)

        reprompt_ids = [_tokenize_reprompt(msgs) for msgs in messages]
        reprompt_elided_flags: list[bool] = []
        reprompt_dropped: list[int] = []
        kept_ids: list[list[int]] = []
        for idx, ids in enumerate(reprompt_ids):
            budget = max_reprompt_len
            if max_teacher_total_len is not None:
                budget = min(budget, max_teacher_total_len - response_lengths[idx])
            budget = max(budget, min_reprompt_len)
            if len(ids) > budget:
                head = budget // 2
                tail = budget - head
                reprompt_dropped.append(len(ids) - budget)
                reprompt_elided_flags.append(True)
                ids = ids[:head] + ids[len(ids) - tail :] if tail else ids[:head]
            else:
                reprompt_dropped.append(0)
                reprompt_elided_flags.append(False)
            kept_ids.append(ids)

        # Left padding, matching self.tokenizer.padding_side set in __init__.
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        width = max((len(ids) for ids in kept_ids), default=0)
        teacher_prompt = {
            "input_ids": torch.tensor(
                [[pad_token_id] * (width - len(ids)) + ids for ids in kept_ids], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [[0] * (width - len(ids)) + [1] * len(ids) for ids in kept_ids], dtype=torch.long
            ),
        }

        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        feedback_used = [
            target_valid_flags[idx]
            and feedback_list[idx] is not None
            and (not feedback_only_without_solution or solution_strs[idx] is None)
            for idx in range(batch_size)
        ]
        self_distillation_mask = torch.tensor(
            [
                target_valid_flags[idx] and (solution_strs[idx] is not None or feedback_used[idx])
                for idx in range(batch_size)
            ],
            dtype=torch.float32,
            device=device,
        )

        unique_uids = set(batch.non_tensor_batch["uid"])
        target_response_lengths = [
            response_lengths[idx] for idx in range(batch_size) if target_valid_flags[idx]
        ]
        metrics = {
            "self_distillation/success_group_fraction": len(
                [uid for uid in unique_uids if len(success_by_uid[uid]) > 0]
            )
            / max(len(unique_uids), 1),
            "self_distillation/success_sample_fraction": sum(s is not None for s in solution_strs) / batch_size,
            "self_distillation/feedback_available_fraction": sum(f is not None for f in feedback_list) / batch_size,
            "self_distillation/feedback_used_fraction": sum(feedback_used) / batch_size,
            "self_distillation/sft_proof_repair_fraction": sum(
                feedback_used[idx]
                and self_distillation_cfg.get("environment_feedback_format", "generic") == "sft_proof_repair"
                for idx in range(batch_size)
            )
            / batch_size,
            "self_distillation/invalid_reward_skipped_fraction": 1.0 - (sum(valid_reward_flags) / batch_size),
            "self_distillation/clipped_response_skipped_fraction": sum(
                valid_reward_flags[idx] and clipped_response_flags[idx] for idx in range(batch_size)
            )
            / batch_size,
            "self_distillation/too_long_response_skipped_fraction": sum(
                valid_reward_flags[idx] and too_long_target_flags[idx] and not clipped_response_flags[idx]
                for idx in range(batch_size)
            )
            / batch_size,
            "self_distillation/target_skipped_fraction": 1.0 - (sum(target_valid_flags) / batch_size),
            # Reprompt shrinkage was previously invisible: apply_chat_template's
            # truncation=True emits no warning and nothing read the prompt width.
            "self_distillation/reprompt_elided_fraction": sum(reprompt_elided_flags) / batch_size,
            "self_distillation/reprompt_tokens_dropped_mean": float(np.mean(reprompt_dropped))
            if reprompt_dropped
            else 0.0,
            "self_distillation/reprompt_tokens_dropped_max": float(max(reprompt_dropped))
            if reprompt_dropped
            else 0.0,
            "self_distillation/teacher_sequence_len_max": float(
                max((len(ids) + response_lengths[idx] for idx, ids in enumerate(kept_ids)), default=0)
            ),
            "self_distillation/target_response_length_mean": float(np.mean(target_response_lengths))
            if target_response_lengths
            else 0.0,
            "self_distillation/target_response_length_max": float(max(target_response_lengths))
            if target_response_lengths
            else 0.0,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
        }
        return (
            DataProto.from_dict(
                tensors={
                    "teacher_input_ids": teacher_input_ids,
                    "teacher_attention_mask": teacher_attention_mask,
                    "teacher_position_ids": teacher_position_ids,
                    "self_distillation_mask": self_distillation_mask,
                }
            ),
            metrics,
        )

    def _proof_action_response_mask(self, batch: DataProto, metrics: dict[str, Any]) -> torch.Tensor | None:
        responses = batch.batch.get("responses", None)
        response_mask = batch.batch.get("response_mask", None)
        if responses is None or response_mask is None:
            return None

        max_response_len = responses.shape[1]
        action_mask = response_mask.clone()
        masked_tokens = 0
        feedback_rows = 0
        mismatch_rows = 0

        responses_cpu = responses.detach().cpu()
        response_lengths = response_mask.detach().sum(dim=1).long().cpu().tolist()
        for row_idx, valid_len in enumerate(response_lengths):
            if valid_len <= 0:
                continue

            token_ids = responses_cpu[row_idx, :valid_len].tolist()
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            spans = _generated_feedback_spans(text)
            if not spans:
                continue

            feedback_rows += 1
            try:
                encoded = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    max_length=max_response_len,
                    truncation=True,
                    return_offsets_mapping=True,
                )
                offsets = list(encoded.get("offset_mapping", []))
            except Exception:
                mismatch_rows += 1
                continue

            if abs(len(offsets) - valid_len) > 2:
                mismatch_rows += 1

            limit = min(valid_len, len(offsets), max_response_len)
            for token_idx in range(limit):
                start, end = offsets[token_idx]
                if end <= start:
                    continue
                if _span_overlaps(start, end, spans):
                    if action_mask[row_idx, token_idx] != 0:
                        masked_tokens += 1
                    action_mask[row_idx, token_idx] = 0

        if feedback_rows:
            denom = max(float(response_mask.sum().detach().item()), 1.0)
            metrics["feedback/generated_feedback_rows"] = feedback_rows
            metrics["feedback/generated_feedback_masked_tokens"] = float(masked_tokens)
            metrics["feedback/generated_feedback_masked_token_rate"] = float(masked_tokens) / denom
            metrics["feedback/generated_feedback_tokenizer_mismatch_rows"] = mismatch_rows

        return action_mask

    def _lean_tail_eos_ids(self) -> frozenset:
        """Ids that legitimately END a generation, used only to carve the EOS out of the cut.

        Built by NAME rather than by guessing an id: an unknown string looked up in a
        vocabulary that does not have it would either KeyError or, worse, resolve to some
        live non-terminal token that then survives every truncation.
        """
        cached = getattr(self, "_lean_tail_eos_ids_cache", None)
        if cached is not None:
            return cached

        ids: set[int] = set()
        raw = getattr(self.tokenizer, "eos_token_id", None)
        candidates = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
        generation_config = getattr(self.tokenizer, "generation_config", None)
        gen_eos = getattr(generation_config, "eos_token_id", None)
        candidates += list(gen_eos) if isinstance(gen_eos, (list, tuple, set)) else [gen_eos]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                ids.add(int(candidate))
            except (TypeError, ValueError):
                continue

        # Qwen3 chat ends a turn on <|im_end|>, which is not always tokenizer.eos_token_id.
        try:
            vocab = self.tokenizer.get_vocab() or {}
        except Exception:
            vocab = {}
        for name in ("<|im_end|>", "<|endoftext|>"):
            if name in vocab:
                ids.add(int(vocab[name]))

        self._lean_tail_eos_ids_cache = frozenset(ids)
        return self._lean_tail_eos_ids_cache

    def _lean_tail_max_trailing_ws(self) -> int:
        """How much whitespace may sit between a closing fence and a force-kept EOS.

        Read from LEAN_MAX_TRAILING_WS, the SAME environment variable the reward manager
        reads for its termination rule, and not from a second Hydra key: the trainer is
        deciding whether a row "ended at its proof", and that question already has an
        owner. A second knob would let the two answers drift, and the drift is silent --
        the reward would stop calling a padded row terminated while the trainer went on
        force-keeping its EOS, or the reverse.
        """
        cached = getattr(self, "_lean_tail_max_trailing_ws_cache", None)
        if cached is not None:
            return cached

        value = _LEAN_MAX_TRAILING_WS_DEFAULT
        raw = os.environ.get("LEAN_MAX_TRAILING_WS", None)
        if raw is not None:
            try:
                parsed = int(str(raw).strip())
            except (TypeError, ValueError):
                parsed = -1
            if parsed < 0:
                logger.warning(
                    "LEAN_MAX_TRAILING_WS=%r is not an integer >= 0; the EOS adjacency rule "
                    "will use %d.",
                    raw,
                    value,
                )
            else:
                value = parsed

        self._lean_tail_max_trailing_ws_cache = value
        return value

    def _lean_tail_response_mask(self, batch: DataProto, metrics: dict[str, Any]) -> torch.Tensor | None:
        """Keep-mask that ends a response ONE token past its last CLOSED lean block; see
        lean_tail_response_mask for the measured reason this exists, and for why that one
        extra token is not an off-by-one.

        Returns None whenever the truncation is off or cannot be done SAFELY, and the
        caller then leaves every mask untouched. Failing open is the whole discipline
        here: a boundary that lands early would drop real proof tokens from the gradient on
        a verified rollout, and nothing downstream would read as anything worse than a run
        that learns slowly.
        """
        # Written unconditionally so a flat line reads as "nothing was truncated" rather
        # than "the metric stopped being emitted".
        metrics["mask/truncated_token_rate"] = 0.0
        metrics["mask/masked_tokens"] = 0.0
        metrics["mask/truncated_rows"] = 0.0
        metrics["mask/truncated_row_rate"] = 0.0
        metrics["mask/no_closed_block_rows"] = 0.0
        metrics["mask/map_fallback_rows"] = 0.0
        # Rows whose terminal EOS was NOT force-kept because it sat behind a tail. This
        # is the population the adjacency rule newly takes gradient away from, so it is
        # the series that says how often the old unconditional carve-out was handing a
        # derailed row a negative gradient on its own stop token.
        metrics["mask/eos_masked_rows"] = 0.0
        # 1.0 only when the cut actually ran this step. A disabled run is otherwise
        # indistinguishable in W&B from a run in which nothing needed truncating.
        metrics["mask/enabled"] = 0.0

        algorithm_cfg = getattr(self.config, "algorithm", None)
        try:
            enabled = algorithm_cfg.get("mask_after_last_lean_block", True)
        except AttributeError:
            enabled = getattr(algorithm_cfg, "mask_after_last_lean_block", True)
        if not _truthy(True if enabled is None else enabled):
            return None

        responses = batch.batch.get("responses", None)
        response_mask = batch.batch.get("response_mask", None)
        if responses is None or response_mask is None:
            return None

        end_bytes = batch.non_tensor_batch.get("lean_last_block_end_byte", None)
        if end_bytes is None or len(end_bytes) != responses.shape[0]:
            # No offsets means a reward manager that does not emit them. Silently doing
            # nothing is correct, but say so once: this is also what a renamed key looks
            # like, and a renamed key would disable the fix without failing anything.
            if not getattr(self, "_lean_tail_missing_key_warned", False):
                self._lean_tail_missing_key_warned = True
                logger.warning(
                    "lean_last_block_end_byte is absent from reward_extra_info; response-mask "
                    "truncation after the last closed lean block is INACTIVE."
                )
            return None

        table = getattr(self, "_lean_tail_byte_table", _UNSET)
        if table is _UNSET:
            table = build_token_byte_lengths(self.tokenizer)
            self._lean_tail_byte_table = table
        if table is None:
            return None

        if not getattr(self, "_lean_tail_byte_table_validated", False):
            lengths = response_mask.detach().sum(dim=1).long().cpu().numpy()
            verdict = token_byte_lengths_agree_with_decode(
                self.tokenizer, table, responses.detach().cpu().numpy(), lengths
            )
            if verdict is False:
                # Not a byte-level BPE vocabulary (or decode does not round-trip). Rather
                # than cut at a guessed token, turn the feature off for the whole run.
                self._lean_tail_byte_table = None
                self._lean_tail_byte_table_validated = True
                logger.warning(
                    "byte-length table does not reproduce the decoded response for %s; "
                    "mask truncation after the last closed lean block is DISABLED for this run.",
                    type(self.tokenizer).__name__,
                )
                return None
            if verdict is None:
                # Nothing in THIS batch could be checked. Truncate nothing this step and
                # ask again next step: disabling a multi-day run because one early batch
                # happened to hold only mid-character or out-of-vocab rows is the worse
                # failure, and it is near-silent (every mask/* metric just stays 0.0).
                attempts = getattr(self, "_lean_tail_validation_attempts", 0) + 1
                self._lean_tail_validation_attempts = attempts
                if attempts >= _LEAN_TAIL_VALIDATION_ATTEMPTS:
                    self._lean_tail_byte_table = None
                    self._lean_tail_byte_table_validated = True
                    logger.warning(
                        "byte-length table could not be validated on any of %d batches for %s; "
                        "mask truncation after the last closed lean block is DISABLED for this run.",
                        attempts,
                        type(self.tokenizer).__name__,
                    )
                else:
                    logger.warning(
                        "byte-length table validation was inconclusive on this batch (attempt %d/%d); "
                        "truncating nothing this step.",
                        attempts,
                        _LEAN_TAIL_VALIDATION_ATTEMPTS,
                    )
                return None
            self._lean_tail_byte_table_validated = True

            # megatron_actor.py reads response_mask and has no ppo_response_mask fallback,
            # so on Megatron this truncation -- like the existing feedback masking and the
            # valid_reward row masking -- is a silent no-op.
            try:
                strategy = str(self.config.actor_rollout_ref.actor.get("strategy", "fsdp"))
            except Exception:
                strategy = "fsdp"
            if "megatron" in strategy.lower():
                logger.warning(
                    "actor strategy is %r; the Megatron actor ignores ppo_response_mask, so the "
                    "truncation after the last closed lean block will NOT reach the loss.",
                    strategy,
                )

        ws_table = getattr(self, "_lean_tail_ws_table", _UNSET)
        if ws_table is _UNSET:
            ws_table = build_token_trailing_ws_bytes(self.tokenizer, table)
            self._lean_tail_ws_table = ws_table
            if ws_table is None:
                # Not fatal, and deliberately not a reason to disable the cut: without it
                # the terminal EOS is simply force-kept wherever it lands, which is the
                # behaviour that shipped before the adjacency rule. Say it once, because
                # otherwise mask/eos_masked_rows reading a flat 0.0 looks like "no row
                # ever rambled past its proof".
                logger.warning(
                    "no trailing-whitespace table could be built for %s; the terminal EOS "
                    "will be force-kept wherever it lands, as before the adjacency rule.",
                    type(self.tokenizer).__name__,
                )

        tail_mask, stats = lean_tail_response_mask(
            responses,
            response_mask,
            end_bytes,
            table,
            eos_ids=self._lean_tail_eos_ids(),
            response_chars=batch.non_tensor_batch.get("response_chars", None),
            token_trailing_ws=ws_table,
            max_trailing_ws=self._lean_tail_max_trailing_ws(),
        )

        rows = max(stats["rows"], 1.0)
        response_tokens = max(stats["response_tokens"], 1.0)
        metrics["mask/truncated_token_rate"] = stats["masked_tokens"] / response_tokens
        # The same quantity unnormalised. The rate alone cannot separate "a few rows
        # rambled for 10k tokens" from "every row carried a short tail", and the two want
        # different responses.
        metrics["mask/masked_tokens"] = stats["masked_tokens"]
        metrics["mask/truncated_rows"] = stats["truncated_rows"]
        metrics["mask/truncated_row_rate"] = stats["truncated_rows"] / rows
        metrics["mask/no_closed_block_rows"] = stats["no_block_rows"]
        # This one should sit at 0. Anything else means the two processes disagree about
        # what a row says, and the truncation is quietly doing nothing on those rows.
        metrics["mask/map_fallback_rows"] = stats["fallback_rows"]
        metrics["mask/eos_masked_rows"] = stats["eos_masked_rows"]
        metrics["mask/enabled"] = 1.0
        return tail_mask

    def _tokenize_aux_target(self, text: str, max_response_len: int) -> tuple[list[int], list[tuple[int, int]], bool]:
        if not text:
            return [], [], False

        # A slow tokenizer cannot return offset mappings. Without them every span
        # maps to nothing, every aux row is dropped for having zero weight, and the
        # entire feedback objective silently becomes a no-op with no error and no
        # warning -- the only tell being feedback/aux_rows reading 0. Fail loudly.
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "feedback_loss requires a fast tokenizer (offset mapping); got "
                f"{type(self.tokenizer).__name__}. Disable actor.feedback_loss.enabled "
                "or load a fast tokenizer."
            )

        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=max_response_len,
                truncation=True,
                return_offsets_mapping=True,
            )
            offsets = list(encoded.get("offset_mapping", []))
            token_ids = list(encoded["input_ids"])
        except Exception:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=max_response_len,
                truncation=True,
            )
            offsets = []
            token_ids = list(encoded["input_ids"])

        truncated = len(token_ids) >= max_response_len and bool(text)
        return token_ids, offsets, truncated

    def _feedback_token_weights(
        self,
        text: str,
        spans,
        error_spans,
        max_response_len: int,
        feedback_token_weight: float = 0.3,
        error_feedback_token_weight: float = 0.5,
    ):
        feedback_weights = torch.zeros(max_response_len, dtype=torch.float32)
        error_weight = torch.zeros(max_response_len, dtype=torch.float32)
        if not text:
            return [], feedback_weights, error_weight, False

        token_ids, offsets, truncated = self._tokenize_aux_target(text, max_response_len)
        limit = len(token_ids)
        for idx in range(limit):
            if idx >= len(offsets):
                break
            start, end = offsets[idx]
            if end <= start:
                continue
            if _span_overlaps(start, end, error_spans):
                feedback_weights[idx] = error_feedback_token_weight
                error_weight[idx] = error_feedback_token_weight
            elif _span_overlaps(start, end, spans):
                feedback_weights[idx] = feedback_token_weight
        return token_ids, feedback_weights, error_weight, truncated

    def _uniform_token_weights(self, text: str, max_response_len: int, weight: float):
        token_ids, _, truncated = self._tokenize_aux_target(text, max_response_len)
        weights = torch.zeros(max_response_len, dtype=torch.float32)
        if token_ids and weight > 0:
            weights[: len(token_ids)] = weight
        return token_ids, weights, truncated

    def _maybe_append_feedback_aux_batch(self, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        # ORDERING DEPENDENCY -- this MUST keep running after compute_advantage.
        # The aux rows are built by index_select from real rows, so each one inherits its
        # source row's uid, lean_status and lean_excess_blocks_penalty while carrying
        # rm_scores == 0 and a zero ppo_response_mask: by content, nothing distinguishes an
        # aux row from a real rollout. Called before the advantage instead, a duplicated
        # "verified" would suppress the all-fail refund for a genuine group, and a
        # duplicated penalty would be refunded onto a zero score. This returns a NEW
        # DataProto rather than mutating `batch`, which is the other half of the guarantee.
        # See compute_grpo_outcome_advantage, which asserts one group id per scored row.
        feedback_cfg = self.config.actor_rollout_ref.actor.get("feedback_loss", {})
        if not _truthy(feedback_cfg.get("enabled", False)):
            return batch

        feedback_token_weight = float(feedback_cfg.get("feedback_weight", 0.3))
        error_feedback_token_weight = float(feedback_cfg.get("error_feedback_weight", 0.5))
        theorem_statement_enabled = _truthy(feedback_cfg.get("theorem_statement_enabled", True))
        theorem_statement_weight = float(feedback_cfg.get("theorem_statement_weight", 0.05))
        # Present the CE target in the same surface form the policy emits at rollout
        # time (prose lead-in + ```lean4 fence). Literal split, not str.format: Lean
        # code is full of braces. Feedback spans are character offsets into the bare
        # annotated code, so they shift by the prefix length.
        if feedback_cfg.get("target_template", None) is None:
            raise ValueError(
                "feedback_loss.target_template is missing from the actor config. Without it the CE "
                "target would silently be emitted unwrapped, in a surface form the policy never "
                "produces. Update the config (verl/trainer/config/actor/actor.yaml)."
            )
        target_template = str(feedback_cfg.get("target_template"))
        if target_template.count("{code}") != 1:
            raise ValueError(
                "feedback_loss.target_template must contain '{code}' exactly once, got "
                f"{target_template!r}"
            )
        target_prefix, target_suffix = target_template.split("{code}")

        canonical_codes = batch.non_tensor_batch.get("canonical_annotated_code", _object_array_1d(["" for _ in range(len(batch))]))
        feedback_spans = batch.non_tensor_batch.get("feedback_spans", _object_array_1d([[] for _ in range(len(batch))]))
        error_spans = batch.non_tensor_batch.get(
            "error_feedback_spans", _object_array_1d([[] for _ in range(len(batch))])
        )
        has_canonical = batch.non_tensor_batch.get(
            "has_canonical_feedback", _object_array_1d([False for _ in range(len(batch))])
        )
        theorem_targets = batch.non_tensor_batch.get(
            "theorem_statement_target", _object_array_1d(["" for _ in range(len(batch))])
        )
        valid_reward_flags = batch.non_tensor_batch.get(
            "lean_valid_reward", _object_array_1d([True for _ in range(len(batch))])
        )

        base = batch.batch
        bsz = len(batch)
        max_response_len = base["responses"].shape[1]
        prompt_len = base["prompts"].shape[1]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        aux_responses: list[torch.Tensor] = []
        aux_response_masks: list[torch.Tensor] = []
        aux_feedback_weights: list[torch.Tensor] = []
        aux_error_weights: list[torch.Tensor] = []
        aux_theorem_weights: list[torch.Tensor] = []
        source_indices: list[int] = []
        feedback_aux_rows = 0
        theorem_aux_rows = 0
        padding_aux_rows = 0
        truncated_feedback_targets = 0
        truncated_theorem_targets = 0
        dropped_feedback_targets = 0

        for idx, (code, spans, err_spans, flag) in enumerate(
            zip(canonical_codes, feedback_spans, error_spans, has_canonical, strict=True)
        ):
            if not _truthy(flag) or not isinstance(code, str) or not code.strip():
                continue
            shift = len(target_prefix)
            token_ids, weights, err_weights, truncated = self._feedback_token_weights(
                f"{target_prefix}{code}{target_suffix}",
                [[int(a) + shift, int(b) + shift] for a, b in (spans or [])],
                [[int(a) + shift, int(b) + shift] for a, b in (err_spans or [])],
                max_response_len,
                feedback_token_weight=feedback_token_weight,
                error_feedback_token_weight=error_feedback_token_weight,
            )
            truncated_feedback_targets += int(truncated)
            if not token_ids or weights.sum().item() <= 0:
                # Every supervised span fell past max_response_len, so this row would
                # contribute nothing. Counting it matters: the increment used to sit
                # after this `continue`, so a target truncated hard enough to lose all
                # its feedback vanished from the objective AND from the metrics.
                dropped_feedback_targets += 1
                continue
            n_tokens = len(token_ids)
            response = torch.full((max_response_len,), int(pad_token_id), dtype=base["responses"].dtype, device=base["responses"].device)
            response_mask = torch.zeros((max_response_len,), dtype=base["response_mask"].dtype, device=base["response_mask"].device)
            response[:n_tokens] = torch.tensor(token_ids, dtype=response.dtype, device=response.device)
            response_mask[:n_tokens] = 1
            aux_responses.append(response)
            aux_response_masks.append(response_mask)
            aux_feedback_weights.append(weights.to(base["responses"].device))
            aux_error_weights.append(err_weights.to(base["responses"].device))
            aux_theorem_weights.append(torch.zeros(max_response_len, dtype=torch.float32, device=base["responses"].device))
            source_indices.append(idx)
            feedback_aux_rows += 1

        if theorem_statement_enabled and theorem_statement_weight > 0:
            for idx, (target, valid_reward) in enumerate(zip(theorem_targets, valid_reward_flags, strict=True)):
                if not _truthy(valid_reward) or not isinstance(target, str) or not target.strip():
                    continue
                token_ids, weights, truncated = self._uniform_token_weights(
                    target, max_response_len, theorem_statement_weight
                )
                if not token_ids or weights.sum().item() <= 0:
                    continue
                truncated_theorem_targets += int(truncated)
                n_tokens = len(token_ids)
                response = torch.full(
                    (max_response_len,),
                    int(pad_token_id),
                    dtype=base["responses"].dtype,
                    device=base["responses"].device,
                )
                response_mask = torch.zeros(
                    (max_response_len,), dtype=base["response_mask"].dtype, device=base["response_mask"].device
                )
                response[:n_tokens] = torch.tensor(token_ids, dtype=response.dtype, device=response.device)
                response_mask[:n_tokens] = 1
                aux_responses.append(response)
                aux_response_masks.append(response_mask)
                aux_feedback_weights.append(torch.zeros(max_response_len, dtype=torch.float32, device=base["responses"].device))
                aux_error_weights.append(torch.zeros(max_response_len, dtype=torch.float32, device=base["responses"].device))
                aux_theorem_weights.append(weights.to(base["responses"].device))
                source_indices.append(idx)
                theorem_aux_rows += 1

        if not aux_responses:
            metrics["feedback/dropped_targets"] = dropped_feedback_targets
            metrics["feedback/truncated_targets"] = truncated_feedback_targets
            metrics["feedback/active_tokens"] = 0.0
            metrics["feedback/theorem_statement_active_tokens"] = 0.0
            return batch

        dp_size = int(self.config.trainer.get("n_gpus_per_node", 1)) * int(self.config.trainer.get("nnodes", 1))
        dp_size = max(dp_size, 1)
        remainder = (bsz + len(aux_responses)) % dp_size
        if remainder:
            padding_aux_rows = dp_size - remainder
            for _ in range(padding_aux_rows):
                aux_responses.append(
                    torch.full(
                        (max_response_len,),
                        int(pad_token_id),
                        dtype=base["responses"].dtype,
                        device=base["responses"].device,
                    )
                )
                aux_response_masks.append(
                    torch.zeros((max_response_len,), dtype=base["response_mask"].dtype, device=base["response_mask"].device)
                )
                aux_feedback_weights.append(
                    torch.zeros(max_response_len, dtype=torch.float32, device=base["responses"].device)
                )
                aux_error_weights.append(
                    torch.zeros(max_response_len, dtype=torch.float32, device=base["responses"].device)
                )
                aux_theorem_weights.append(
                    torch.zeros(max_response_len, dtype=torch.float32, device=base["responses"].device)
                )
                source_indices.append(0)

        aux_responses_tensor = torch.stack(aux_responses, dim=0)
        aux_response_mask = torch.stack(aux_response_masks, dim=0)
        feedback_weights = torch.stack(aux_feedback_weights, dim=0)
        error_weights = torch.stack(aux_error_weights, dim=0)
        theorem_weights = torch.stack(aux_theorem_weights, dim=0)
        combined_ce_weights = feedback_weights + theorem_weights

        source_index = torch.tensor(source_indices, dtype=torch.long)
        prompt_attention = base["attention_mask"].index_select(
            0, source_index.to(base["attention_mask"].device)
        )[:, :prompt_len]
        aux_attention_mask = torch.cat([prompt_attention, aux_response_mask], dim=1)
        aux_prompts = base["prompts"].index_select(0, source_index.to(base["prompts"].device))
        aux_input_ids = torch.cat([aux_prompts, aux_responses_tensor], dim=1)
        aux_position_ids = compute_position_id_with_mask(aux_attention_mask)
        if base["position_ids"].dim() == 3:
            aux_position_ids = base["position_ids"].index_select(0, source_index.to(base["position_ids"].device)).clone()

        tensors = {}
        for key, tensor in base.items():
            if key == "prompts":
                aux_tensor = tensor.index_select(0, source_index.to(tensor.device)).clone()
            elif key == "responses":
                aux_tensor = aux_responses_tensor
            elif key == "input_ids":
                aux_tensor = aux_input_ids
            elif key == "attention_mask":
                aux_tensor = aux_attention_mask
            elif key == "position_ids":
                aux_tensor = aux_position_ids
            elif key == "response_mask":
                aux_tensor = aux_response_mask
            elif key in ("teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"):
                # SDPO only. These are built BEFORE this function runs (fit() calls
                # _maybe_build_self_distillation_batch first), so aux rows have no teacher
                # tensors of their own. They must NOT be zeros: with use_remove_padding an
                # all-zero attention mask unpads to ZERO tokens (cu_seqlens=[0,0]), and that
                # degenerate forward is at best wasted work and at worst a NaN the later
                # self_distillation_mask cannot undo (masking is applied after the fact, and
                # 0 * NaN = NaN).
                #
                # Copying the source rollout's teacher tensors keeps the forward well-formed.
                # Note we deliberately do NOT skip the teacher forward for these rows instead:
                # it runs through the FSDP-wrapped module, so it is a collective, and rows are
                # re-partitioned across ranks by sequence length
                # (_balance_batch -> get_seqlen_balanced_partitions). At
                # ppo_micro_batch_size_per_gpu=1 one rank could then skip a forward at the same
                # loop index where another rank enters it, and the run deadlocks on the
                # all-gather. Uniform control flow is worth the redundant compute.
                aux_tensor = tensor.index_select(0, source_index.to(tensor.device)).clone()
            else:
                # self_distillation_mask lands here and MUST stay zero: that is what excludes
                # aux rows from the distillation loss. lean_tail_mask lands here too and the
                # zeros are harmless -- it is consumed by compute_advantage, which has
                # already run, and it is not in the actor's select_keys.
                aux_tensor = torch.zeros_like(tensor.index_select(0, source_index.to(tensor.device)))
            tensors[key] = torch.cat([tensor, aux_tensor], dim=0)

        # Must carry forward the EXISTING ppo_response_mask, not response_mask: by this
        # point it already encodes two deliberate maskings applied in fit() --
        # _proof_action_response_mask (the model's own generated <feedback> blocks are
        # excluded from the PPO objective) and valid_reward_mask (rows whose Lean reward
        # was invalid are zeroed). Rebuilding it from response_mask silently discarded
        # both, and only on steps that happened to produce aux rows.
        tensors["ppo_response_mask"] = torch.cat(
            [base.get("ppo_response_mask", base["response_mask"]), torch.zeros_like(aux_response_mask)], dim=0
        )
        tensors["feedback_loss_weights"] = torch.cat(
            [torch.zeros((bsz, max_response_len), dtype=torch.float32, device=combined_ce_weights.device), combined_ce_weights],
            dim=0,
        )
        tensors["error_feedback_loss_weights"] = torch.cat(
            [torch.zeros((bsz, max_response_len), dtype=torch.float32, device=error_weights.device), error_weights], dim=0
        )
        tensors["theorem_statement_loss_weights"] = torch.cat(
            [torch.zeros((bsz, max_response_len), dtype=torch.float32, device=theorem_weights.device), theorem_weights],
            dim=0,
        )

        non_tensors = {
            key: np.concatenate([value, np.asarray(value, dtype=object)[source_indices]], axis=0)
            for key, value in batch.non_tensor_batch.items()
        }
        metrics["feedback/aux_rows"] = len(aux_responses)
        metrics["feedback/canonical_aux_rows"] = feedback_aux_rows
        metrics["feedback/theorem_statement_aux_rows"] = theorem_aux_rows
        metrics["feedback/padding_aux_rows"] = padding_aux_rows
        metrics["feedback/active_tokens"] = float((feedback_weights > 0).sum().item())
        metrics["feedback/error_active_tokens"] = float((error_weights > 0).sum().item())
        metrics["feedback/theorem_statement_active_tokens"] = float((theorem_weights > 0).sum().item())
        metrics["feedback/avg_feedback_tokens"] = float((feedback_weights > 0).sum().item() / max(feedback_aux_rows, 1))
        metrics["feedback/avg_error_feedback_tokens"] = float((error_weights > 0).sum().item() / max(feedback_aux_rows, 1))
        metrics["feedback/avg_theorem_statement_tokens"] = float(
            (theorem_weights > 0).sum().item() / max(theorem_aux_rows, 1)
        )
        metrics["feedback/truncated_targets"] = truncated_feedback_targets
        metrics["feedback/dropped_targets"] = dropped_feedback_targets
        metrics["feedback/theorem_statement_truncated_targets"] = truncated_theorem_targets

        # Deal the rows round-robin across the DP ranks. DataProto.chunk() splits
        # contiguously, and the aux rows are all appended at the end, so without this
        # the last ranks receive only aux rows (zero PPO gradient) and the first ranks
        # only real rows (zero feedback gradient). At the observed ~87% canonical-feedback
        # rate that is a clean 4/4 split of an 8-rank job, and since FSDP averages
        # gradients across ranks it silently halves BOTH objectives -- by a factor that
        # drifts step to step with how many rows produced canonical feedback.
        total_rows = bsz + len(aux_responses)
        if dp_size > 1 and total_rows % dp_size == 0:

            def _interleave(rows: list[int]) -> list[int]:
                """Blend rollout and aux rows proportionally within one rank's slice.

                Dealing rows round-robin balances the RANKS, but each rank still
                receives its rollout rows before its aux rows. update_policy splits a
                rank's rows into mini-batches and takes an OPTIMIZER STEP PER
                MINI-BATCH, so an unmixed slice puts RL+KL in one step and the feedback
                CE in another. That is not the paper's summed objective, and because
                Adam renormalises each step, lambda_coef loses its meaning entirely.

                Verified by sweep: with one aux row per rollout row (theorem-statement CE
                off, so n_aux <= bsz) no mini-batch is ever CE-only. Enabling
                theorem_statement_enabled can push n_aux above bsz, where a short trailing
                mini-batch can again become CE-only; re-check the split if you turn it on.
                """
                rollout_rows = [row for row in rows if row < bsz]
                aux_rows = [row for row in rows if row >= bsz]
                if not rollout_rows or not aux_rows:
                    return rows
                keyed = [(idx / len(rollout_rows), 0, row) for idx, row in enumerate(rollout_rows)]
                keyed += [(idx / len(aux_rows), 1, row) for idx, row in enumerate(aux_rows)]
                keyed.sort()
                return [row for _, _, row in keyed]

            order = [
                row
                for start in range(dp_size)
                for row in _interleave(list(range(start, total_rows, dp_size)))
            ]
            index = torch.tensor(order, dtype=torch.long)
            tensors = {key: value.index_select(0, index.to(value.device)) for key, value in tensors.items()}
            order_np = np.asarray(order)
            non_tensors = {key: value[order_np] for key, value in non_tensors.items()}

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=deepcopy(batch.meta_info))

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        metrics.update(self._lean_reward_diagnostics(reward_tensor, reward_extra_infos_dict))

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            # Emitted under BOTH names, deliberately, and they are the same
                            # number here because this aggregation uses response_mask -- the
                            # WHOLE generated response, computed before compute_advantage
                            # narrows anything (ppo_response_mask does not exist yet at this
                            # point in the loop).
                            #
                            # actor/entropy_full_response has to exist unconditionally. The
                            # entropy collapse that ended qwen3-sft-feedback-grpo-lr2e6 went
                            # 0.019 -> 3.5 with 75-90% of the excursion living in the tail
                            # that ppo_response_mask now removes, so a detector confined to
                            # the kept prefix can miss the next one. dp_actor.py computes
                            # both masks properly, but only inside `if calculate_entropy`,
                            # which is OFF by default -- so on a default run that name never
                            # appeared and the documented tripwire had no series at all.
                            # When calculate_entropy IS on, dp_actor overrides both keys
                            # with its own values and the meanings still hold: actor/entropy
                            # becomes the narrowed view, actor/entropy_full_response stays
                            # the whole response.
                            entropy_value = entropy_agg.detach().item()
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_value,
                                "actor/entropy_full_response": entropy_value,
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        self_distillation_data = self._maybe_build_self_distillation_batch(
                            batch, reward_tensor, reward_extra_infos_dict
                        )
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: _object_array_1d(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        proof_action_mask = self._proof_action_response_mask(batch, metrics)
                        if proof_action_mask is not None:
                            batch.batch["ppo_response_mask"] = proof_action_mask

                        # Third narrowing factor on the PPO mask, after the generated
                        # <feedback> spans and before the valid_reward row zeroing. Kept as
                        # its own tensor so compute_advantage can use the tail cut alone,
                        # and composed MULTIPLICATIVELY so the order of these three is
                        # irrelevant -- assigning would clobber whichever ran first.
                        #
                        # Position matters twice: this is AFTER
                        # _maybe_build_self_distillation_batch, so the SDPO teacher tensors
                        # were built from an untouched response_mask (a holed
                        # teacher_attention_mask over full teacher_input_ids renumbers
                        # positions and, with use_remove_padding, drops tokens the student
                        # still scores); and BEFORE _maybe_append_feedback_aux_batch, so the
                        # aux rows inherit the composed mask instead of a rebuilt one.
                        lean_tail_mask = self._lean_tail_response_mask(batch, metrics)
                        if lean_tail_mask is not None:
                            batch.batch["lean_tail_mask"] = lean_tail_mask
                            base_ppo_mask = batch.batch.get("ppo_response_mask", batch.batch["response_mask"])
                            batch.batch["ppo_response_mask"] = base_ppo_mask * lean_tail_mask.to(base_ppo_mask.dtype)

                        valid_reward_mask = _lean_valid_reward_mask(
                            reward_extra_infos_dict,
                            reward_tensor.shape[0],
                            reward_tensor.device,
                        )
                        if valid_reward_mask is not None:
                            batch.batch["valid_reward_mask"] = valid_reward_mask
                            row_mask = valid_reward_mask.to(batch.batch["response_mask"].dtype).unsqueeze(-1)
                            base_ppo_mask = batch.batch.get("ppo_response_mask", batch.batch["response_mask"])
                            batch.batch["ppo_response_mask"] = base_ppo_mask * row_mask
                            metrics["lean/invalid_reward_masked_rate"] = (
                                1.0 - valid_reward_mask.float().mean().detach().item()
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        metrics.update(batch.meta_info.pop("lean_attempt_neutralize_stats", {}))

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_batch = self._maybe_append_feedback_aux_batch(batch, metrics)
                            actor_output = self._update_actor(actor_batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
