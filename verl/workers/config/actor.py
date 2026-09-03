# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig, VeOmniEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "PolicyLossConfig",
    "FeedbackLossConfig",
    "SftReplayConfig",
    "SelfDistillationConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
    "VeOmniActorConfig",
]


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for SDPO self-distillation."""

    full_logit_distillation: bool = True
    loss_coef: float = 1.0
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    distillation_topk: Optional[int] = 20
    distillation_add_tail: bool = True
    distillation_tail_epsilon: float = 1e-4
    skip_clipped_responses: bool = False
    max_target_response_len: Optional[int] = None
    max_reprompt_len: int = 4096
    # Budget the whole teacher sequence (reprompt + response) rather than the
    # reprompt alone; None keeps the legacy reprompt-only behaviour. Over-budget
    # reprompts are elided in the middle, preserving head and tail.
    max_teacher_total_len: Optional[int] = None
    min_reprompt_len: int = 1024
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
    solution_template: str = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
    feedback_template: str = (
        "\nThe following is Lean feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    environment_feedback_format: str = "sft_proof_repair"
    # Mirrors the SFT proof-repair prompt; see the comment in
    # verl/trainer/config/actor/actor.yaml. Keep the two in sync.
    proof_repair_template: str = (
        "{prompt}\n\n"
        "Here is the complete Lean 4 proof annotated with Lean 4 compiler feedback blocks:\n\n"
        "```lean4\n{failed_attempt}\n```\n"
    )
    # Give rows with no canonical Lean annotation a status-derived teacher context;
    # see _lean_status_fallback_feedback and the note in actor.yaml.
    use_fallback_environment_feedback: bool = False
    include_environment_feedback: bool = True
    environment_feedback_only_without_solution: bool = True

    def __post_init__(self):
        if self.loss_coef < 0.0:
            raise ValueError(f"self_distillation.loss_coef must be non-negative, got {self.loss_coef}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 < self.distillation_tail_epsilon < 1.0:
            raise ValueError(
                "self_distillation.distillation_tail_epsilon must be in (0, 1), "
                f"got {self.distillation_tail_epsilon}"
            )
        if self.max_teacher_total_len is not None and self.max_teacher_total_len <= 0:
            raise ValueError(
                "self_distillation.max_teacher_total_len must be a positive integer or null, "
                f"got {self.max_teacher_total_len}"
            )
        if self.min_reprompt_len <= 0:
            raise ValueError(
                f"self_distillation.min_reprompt_len must be a positive integer, got {self.min_reprompt_len}"
            )
        if self.max_target_response_len is not None and self.max_target_response_len <= 0:
            raise ValueError(
                "self_distillation.max_target_response_len must be a positive integer or null, "
                f"got {self.max_target_response_len}"
            )
        if self.environment_feedback_format not in {"generic", "sft_proof_repair"}:
            raise ValueError(
                "self_distillation.environment_feedback_format must be one of "
                f"{{'generic', 'sft_proof_repair'}}, got {self.environment_feedback_format!r}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class FeedbackLossConfig(BaseConfig):
    """Auxiliary supervised loss for canonical Lean feedback blocks."""

    enabled: bool = False
    lambda_coef: float = 1.0
    feedback_weight: float = 0.3
    error_feedback_weight: float = 0.5
    theorem_statement_enabled: bool = True
    theorem_statement_weight: float = 0.05
    # Surface form of the CE target. The auxiliary row must present the annotated
    # proof the same way the policy emits it at rollout time, otherwise feedback
    # prediction is trained in a context the policy never occupies -- and the
    # objective quietly pushes it to drop the fence its own reward path requires.
    # Must contain "{code}" exactly once; substitution is literal (Lean code is
    # full of braces, so str.format cannot be used).
    target_template: str = (
        " Here is the complete Lean 4 proof annotated with Lean 4 compiler feedback blocks:"
        "\n\n```lean4\n{code}\n```\n"
    )

    def __post_init__(self):
        if self.target_template.count("{code}") != 1:
            raise ValueError(
                "feedback_loss.target_template must contain '{code}' exactly once, got "
                f"{self.target_template!r}"
            )
        if self.lambda_coef < 0.0:
            raise ValueError(f"feedback_loss.lambda_coef must be non-negative, got {self.lambda_coef}")


@dataclass
class SftReplayConfig(BaseConfig):
    """Supervised replay mixed into the actor update to anchor an RL-eroded behaviour.

    Distinct from FeedbackLossConfig: that reweights tokens of the ROLLOUTS and so can only
    shape how a rollout is written. Replay forwards separate SFT sequences, which is what it
    takes to hold a *decision* (here: retry after a failed proof) rather than a surface form.

    `files` is a PRE-TOKENISED parquet from prepare_sft_replay.py -- input_ids / prompt_len /
    proof_repair. It is tokenizer-specific; a pool built for another model is silently wrong,
    so there is no default path and `enabled` without `files` is refused rather than ignored.
    """

    enabled: bool = False
    lambda_coef: float = 0.1
    # Optional[str], NOT str. grpo.sh passes files="" on the disabled branch, and an empty
    # Hydra override (files=) is parsed as None -- against a plain `str` field that raises
    # ValidationError at config load and kills the launch before a single GPU is touched.
    # Every run with replay OFF, including the whole classic arm, would have died there.
    files: Optional[str] = None
    # GLOBAL per optimizer step; dp_actor divides by the data-parallel world size.
    # 128 = the 16/rank x 8 ranks of the run that demonstrably held repair -- kept as the
    # default in the units that survive a change of GPU count.
    samples_per_step: int = 128
    micro_batch_size: int = 2
    seed: int = 1234

    def __post_init__(self):
        if self.lambda_coef < 0.0:
            raise ValueError(f"sft_replay.lambda_coef must be non-negative, got {self.lambda_coef}")
        # The shape guards apply only when the feature is ON. A launcher that disables replay
        # reasonably writes 0 across every knob to say so, and rejecting that killed the
        # no-replay arm at config load -- before a GPU was touched -- for a value the code
        # never reads. Same conditional shape as the `files` check below.
        if self.enabled and self.micro_batch_size < 1:
            raise ValueError(f"sft_replay.micro_batch_size must be >= 1, got {self.micro_batch_size}")
        if self.enabled and self.samples_per_step < 1:
            raise ValueError(
                f"sft_replay.enabled=true requires samples_per_step >= 1, got {self.samples_per_step}"
            )
        if self.enabled and not self.files:
            raise ValueError(
                "sft_replay.enabled=true requires sft_replay.files. Silently training with no "
                "replay pool would look identical to a working run in every logged metric."
            )


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    feedback_loss: FeedbackLossConfig = field(default_factory=FeedbackLossConfig)
    sft_replay: SftReplayConfig = field(default_factory=SftReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )


@dataclass
class VeOmniActorConfig(ActorConfig):
    """Configuration for VeOmni actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'veomni' for VeOmni parallelism.
        veomni (dict[str, Any]): Configuration for VeOmni settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "veomni"
    veomni: VeOmniEngineConfig = field(default_factory=VeOmniEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate VeOmni actor configuration parameters."""
        super().__post_init__()
        self.engine = self.veomni
