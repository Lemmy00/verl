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

    assert metrics["lean/valid_proof_rate"] == pytest.approx(1 / 3)
    assert metrics["lean/infra_failure_rate"] == pytest.approx(1 / 3)
    assert metrics["lean/error_kind/wall_timeout"] == 1
    assert metrics["lean/error_kind_rate/compiler_error"] == pytest.approx(1 / 3)
    assert metrics["lean/valid_reward_rate"] == pytest.approx(2 / 3)
    assert metrics["lean/timeout_events"] == 1.0
    assert metrics["lean/timeout_rollout_rate"] == pytest.approx(1 / 3)
    assert metrics["lean/candidate_timeout_events"] == 1.0
    assert metrics["lean/retry_events"] == 1.0
    assert metrics["lean/command_attempt_events"] == 4.0
    assert metrics["lean/context_wait_s/mean"] == 2.0
    assert metrics["lean/verify_wall_s/p90"] == pytest.approx(49.6)
    assert metrics["lean/total_wall_s/max"] == 62.0
    assert metrics["lean/problem_cache_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["lean/context_cache_or_env_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["lean/executor_workers"] == 64.0
    assert metrics["lean/warmup_attempts_total"] == 65.0
    assert metrics["lean/restart_warmups_total"] == 1.0
    assert metrics["lean/loss_reward_mean"] == pytest.approx(1 / 3)
    assert metrics["lean/reward_mean"] == pytest.approx(1 / 3)
