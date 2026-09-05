# Copyright 2026 Individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Construction diagnostics preserve verified exemptions and actual GRPO advantages."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _lean_construction_metrics


def _batch(advantages, blocks, verified, **extra):
    return DataProto.from_dict(
        tensors={
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "response_mask": torch.ones(len(advantages), len(advantages[0])),
        },
        non_tensors={
            key: np.array(values, dtype=object)
            for key, values in {
                "lean_closed_blocks": blocks,
                "lean_verified": verified,
                **extra,
            }.items()
        },
    )


def test_post_refund_means_do_not_dilute_long_masked_tails():
    # Only construction/nontermination differentiate this all-fail group; the 0.18
    # effort penalty is refunded on a local copy inside the actual estimator.
    score = torch.tensor([[-0.38, 0, 0, 0], [-0.15, 0, 0, 0]])
    broadcast_mask = torch.tensor([[1.0, 0, 0, 0], [1.0, 1, 1, 1]])
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=score,
        response_mask=broadcast_mask,
        index=np.array(["same", "same"]),
        config=OmegaConf.create({"grpo_adv_std_floor": 0.3535530370398831}),
        attempt_penalty=torch.tensor([0.18, 0.0]),
        verified=torch.tensor([False, False]),
    )
    batch = _batch(advantages.tolist(), [2, 2], [False, False], lean_construction_detected=[True, False])
    batch.batch["lean_tail_mask"] = broadcast_mask
    metrics = RayPPOTrainer._lean_advantage_diagnostics(None, batch)
    expected = float(advantages[0, 0])
    assert expected == pytest.approx(-0.07071068, abs=2e-6)
    assert metrics["lean/adv/construction_failed"] == pytest.approx(expected)
    assert metrics["lean/adv/repair_2_4_failed"] == pytest.approx(0.0, abs=2e-6)
    assert torch.equal(batch.batch["advantages"], advantages)
    assert score.sum().item() == pytest.approx(-0.53)  # diagnostics/estimator do not rewrite dump scores


def test_class_outcome_splits_keep_verified_construction_visible():
    batch = _batch(
        [[0.9, 0.9], [-0.4, -0.4], [-0.6, -0.6], [99.0, 99.0]],
        [2, 2, 6, 2],
        [True, False, False, False],
        lean_valid_reward=[True, True, True, False],
        lean_construction_detected=[True, True, False, True],
        lean_duplicate_detected=[True, False, False, True],
        lean_construction_uncertain=[False, True, False, True],
    )
    metrics = RayPPOTrainer._lean_advantage_diagnostics(None, batch)
    assert metrics["lean/adv/repair_2_4_n"] == 2
    assert metrics["lean/adv/repair_2_4_verified"] == pytest.approx(0.9)
    assert metrics["lean/adv/repair_2_4_failed"] == pytest.approx(-0.4)
    assert metrics["lean/adv/repair_5plus_failed"] == pytest.approx(-0.6)
    assert metrics["lean/adv/construction_verified"] == pytest.approx(0.9)
    assert metrics["lean/adv/construction_failed"] == pytest.approx(-0.4)
    assert metrics["lean/adv/duplicate_failed_n"] == 0
    assert "lean/adv/duplicate_failed" not in metrics
    assert metrics["lean/adv/construction_uncertain_failed_n"] == 1
    assert metrics["lean/adv/one_shot_verified_n"] == 0


def test_detection_charging_and_uncertainty_have_separate_outcome_denominators():
    metrics = _lean_construction_metrics({
        "lean_verified": [True, False, False, False],
        "lean_valid_reward": [True, True, True, False],
        "lean_construction_detected": [True, True, False, True],
        "lean_duplicate_detected": [True, False, False, True],
        "lean_construction_uncertain": [False, False, True, True],
        "lean_construction_penalty": [0.0, 0.2, 0.0, 0.0],
    })
    assert metrics["lean/construction/detected_rate"] == pytest.approx(2 / 3)
    assert metrics["lean/construction/charged_rate"] == pytest.approx(1 / 3)
    assert metrics["lean/construction/verified_detected_rate"] == 1.0
    assert metrics["lean/construction/verified_charged_rate"] == 0.0
    assert metrics["lean/construction/verified_penalty_mean"] == 0.0
    assert metrics["lean/construction/failed_detected_rate"] == 0.5
    assert metrics["lean/construction/failed_penalty_mean"] == pytest.approx(0.1)
    assert metrics["lean/duplicate/verified_detected_rate"] == 1.0
    assert metrics["lean/construction/failed_uncertain_rate"] == 0.5


def test_reward_aggregator_calls_construction_diagnostics():
    metrics = RayPPOTrainer._lean_reward_diagnostics(None, torch.zeros(1, 2), {
        "lean_verified": [True],
        "lean_construction_detected": [True],
        "lean_construction_penalty": [0.0],
    })
    assert metrics["lean/construction/verified_detected_n"] == 1
    assert metrics["lean/construction/verified_charged_n"] == 0
    assert metrics["lean/construction/failed_detected_n"] == 0
    assert "lean/construction/failed_detected_rate" not in metrics


def test_missing_and_short_metadata_do_not_invent_clean_rows():
    assert _lean_construction_metrics({}) == {}
    assert _lean_construction_metrics({
        "lean_verified": [False, True], "lean_construction_detected": [True],
    }) == {}
    assert _lean_construction_metrics({
        "lean_verified": [False, True], "lean_valid_reward": [False],
        "lean_construction_detected": [True, True],
    }) == {}
    batch = _batch([[0.1], [0.2]], [1, 2], [True, False])
    batch.non_tensor_batch["lean_valid_reward"] = np.array([True], dtype=object)
    assert RayPPOTrainer._lean_advantage_diagnostics(None, batch) == {}
