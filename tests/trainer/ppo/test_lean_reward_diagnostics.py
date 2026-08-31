import pytest
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_lean_reward_diagnostics_aggregates_timeout_causes_and_latency():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    extra = {
        "lean_status": ["verified", "lean_infra_failed", "invalid_proof"],
        "lean_error_kind": ["", "wall_timeout", "compiler_error"],
        "lean_valid_reward": [True, False, True],
        "lean_score": [1.0, 0.0, 0.0],
        "lean_timeouts": [0, 1, 0],
        "lean_candidate_timeouts": [0, 1, 0],
        "lean_feedback_fallback_timeouts": [0, 0, 0],
        "lean_setup_timeouts": [0, 0, 0],
        "lean_replay_timeouts": [0, 0, 0],
        "lean_replay_failures": [0, 0, 0],
        "lean_retries": [0, 1, 0],
        "lean_command_attempts": [1, 2, 1],
        "lean_context_s": [1.0, 2.0, 3.0],
        "lean_verify_s": [4.0, 60.0, 8.0],
        "lean_total_s": [5.0, 62.0, 11.0],
        "reward_remote_s": [5.5, 62.5, 11.5],
        "lean_cache_hit": [True, False, True],
        "lean_context_cache_hit": [True, True, False],
        "lean_executor_workers": [64, 64, 64],
        "lean_warmup_attempts_total": [64, 65, 65],
        "lean_warmup_failures_total": [0, 0, 0],
        "lean_restart_warmups_total": [0, 1, 1],
        "lean_restart_warmup_failures_total": [0, 0, 0],
    }

    metrics = trainer._lean_reward_diagnostics(rewards, extra)

    assert metrics["lean/status_rate/verified"] == pytest.approx(1 / 3)
    assert metrics["lean/infra_failure_rate"] == pytest.approx(1 / 3)
    assert metrics["lean/error_kind/wall_timeout"] == 1
    assert metrics["lean/error_kind_rate/compiler_error"] == pytest.approx(1 / 3)
    assert metrics["lean/valid_reward_rate"] == pytest.approx(2 / 3)
    assert metrics["lean/timeout_rate"] == pytest.approx(1 / 3)
    # A candidate timeout IS the lean_timeout status; the duplicate family is gone.
    assert metrics["lean/status_rate/lean_timeout"] == pytest.approx(0.0)
    assert metrics["lean/retry_rate"] == pytest.approx(1 / 3)
    assert metrics["lean/command_attempt_rate"] == pytest.approx(1.0)
    assert metrics["lean/command_attempt_per_rollout"] == pytest.approx(4 / 3)
    assert metrics["lean/context_wait_s/mean"] == 2.0
    assert metrics["lean/verify_wall_s/p90"] == pytest.approx(49.6)
    assert metrics["lean/total_wall_s/max"] == 62.0
    assert metrics["lean/problem_cache_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["lean/context_cache_or_env_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["lean/executor_workers"] == 64.0
    # The warmup gauges moved: this batch has a non-zero infra roll-up, so they are all
    # emitted individually as well.
    assert metrics["lean/infra_events_total"] == pytest.approx(66.0)
    assert metrics["lean/warmup_attempts_total"] == 65.0
    assert metrics["lean/restart_warmups_total"] == 1.0
    assert metrics["lean/reward_mean"] == pytest.approx(1 / 3)
    for dropped in (
        "lean/valid_proof_rate",
        "lean/loss_reward_mean",
        "lean/error_kind_rate/wall_timeout",
        "lean/candidate_timeout_events",
        "lean/candidate_timeout_per_rollout",
        "lean/candidate_timeout_rollout_rate",
        "lean/timeout_events",
        "lean/timeout_rollout_rate",
        "lean/timeout_per_rollout",
        "lean/retry_events",
        "lean/retry_rollout_rate",
        "lean/command_attempt_events",
        "lean/command_attempt_rollout_rate",
    ):
        assert dropped not in metrics


def test_lean_reward_diagnostics_separates_attempts_from_closed_blocks():
    """The fence dodge -- N openers, one closing fence -- is invisible in every metric
    that predates it: lean/attempts_mean, lean/repair_rate and lean/penalized_rate all
    FALL as it spreads, so the batch reads as a policy that learned to be concise.
    lean/unclosed_block_rate is the only series that rises."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    rewards = torch.zeros(4, 2)
    extra = {
        "lean_status": ["verified", "invalid_proof", "invalid_proof", "no_lean_code"],
        # Row 1 is the dodge: eight attempts, one closed block.
        "lean_closed_blocks": [1, 1, 8, 0],
        "lean_block_attempts": [1, 8, 8, 3],
        "lean_block_comment_chars": [0, 0, 40, 0],
        # Of row 2's 40 comment characters, 4 are not /- <feedback> -/ annotations.
        "lean_block_other_comment_chars": [0, 0, 4, 0],
        "lean_block_body_chars": [30, 30, 100, 0],
    }

    metrics = trainer._lean_reward_diagnostics(rewards, extra)

    assert metrics["lean/attempts_mean"] == pytest.approx(10 / 4)
    assert metrics["lean/attempt_blocks_mean"] == pytest.approx(20 / 4)
    assert metrics["lean/unclosed_blocks_mean"] == pytest.approx(10 / 4)
    # Rows 1 and 3 abandoned an opener; rows 0 and 2 closed everything they opened.
    assert metrics["lean/unclosed_block_rate"] == pytest.approx(2 / 4)
    assert metrics["lean/block_comment_chars_mean"] == pytest.approx(10.0)
    assert metrics["lean/block_other_comment_chars_mean"] == pytest.approx(1.0)
    assert metrics["lean/block_body_chars_mean"] == pytest.approx(40.0)


def test_lean_reward_diagnostics_separates_feedback_comments_from_the_rest():
    """The total cannot see rambling move inside the fence. Measured over steps
    110/116/120 of qwen3-sft-feedback-grpo-lr2e6, 98.5% of in-fence comment characters
    (15,839,001 of 16,079,033) are the /- <feedback> -/ blocks the SFT format requires,
    so a doubling of genuine rambling moves lean/block_comment_chars_mean by 1.5% and
    hides in the step-to-step wobble of feedback volume. The residue starts near zero,
    so its slope is readable -- and the two series move independently, which is why both
    are emitted."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    rewards = torch.zeros(2, 2)

    # More feedback, no rambling: the total moves, the residue does not.
    feedback_heavy = trainer._lean_reward_diagnostics(
        rewards,
        {
            "lean_block_comment_chars": [1000, 3000],
            "lean_block_other_comment_chars": [0, 0],
        },
    )
    assert feedback_heavy["lean/block_comment_chars_mean"] == pytest.approx(2000.0)
    assert feedback_heavy["lean/block_other_comment_chars_mean"] == pytest.approx(0.0)

    # The same feedback volume with rambling added: a 1.5% move in the total that would
    # be unreadable on its own, and a residue that went from nothing to something.
    rambling = trainer._lean_reward_diagnostics(
        rewards,
        {
            "lean_block_comment_chars": [1015, 3045],
            "lean_block_other_comment_chars": [15, 45],
        },
    )
    total_move = (
        rambling["lean/block_comment_chars_mean"]
        / feedback_heavy["lean/block_comment_chars_mean"]
        - 1.0
    )
    assert total_move == pytest.approx(0.015)
    assert rambling["lean/block_other_comment_chars_mean"] == pytest.approx(30.0)


def test_lean_reward_diagnostics_skips_new_series_when_the_keys_are_absent():
    """A reward manager that does not emit them must produce NO metric rather than a
    misleading 0.0 -- a missing panel is the signal."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    rewards = torch.zeros(2, 2)
    metrics = trainer._lean_reward_diagnostics(rewards, {"lean_closed_blocks": [1, 2]})

    assert "lean/attempts_mean" in metrics
    for absent in (
        "lean/attempt_blocks_mean",
        "lean/unclosed_block_rate",
        "lean/unclosed_blocks_mean",
        "lean/block_comment_chars_mean",
        "lean/block_other_comment_chars_mean",
        "lean/block_body_chars_mean",
    ):
        assert absent not in metrics
