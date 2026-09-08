# Copyright 2026 Individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""An explicitly selected complete checkpoint must not depend on the latest marker."""

from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _trainer(root, mode, checkpoint):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {
        "resume_mode": mode,
        "resume_from_path": str(checkpoint),
        "default_local_dir": str(root),
        "default_hdfs_dir": None,
        "del_local_ckpt_after_load": False,
    }})
    trainer.global_steps = 0
    trainer.use_critic = False
    trainer.actor_rollout_wg = Mock()
    trainer.train_dataloader = Mock()
    return trainer


@pytest.mark.parametrize("marker", [None, "", "unfinished", "999"])
def test_explicit_resume_ignores_missing_interrupted_or_stale_marker(tmp_path, marker):
    checkpoint = tmp_path / "global_step_100"
    checkpoint.mkdir()
    data_state = {"epoch": 1, "batches_yielded": 100}
    torch.save(data_state, checkpoint / "data.pt")
    if marker is not None:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(marker)

    trainer = _trainer(tmp_path, "resume_path", checkpoint)
    trainer._load_checkpoint()

    assert trainer.global_steps == 100
    trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(
        str(checkpoint / "actor"), del_local_after_load=False
    )
    trainer.train_dataloader.load_state_dict.assert_called_once_with(data_state)


def test_auto_resume_still_uses_its_marker(tmp_path):
    checkpoint = tmp_path / "global_step_90"
    checkpoint.mkdir()
    torch.save({"batches_yielded": 90}, checkpoint / "data.pt")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("90")
    trainer = _trainer(tmp_path, "auto", tmp_path / "global_step_100")

    trainer._load_checkpoint()

    assert trainer.global_steps == 90
    trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(
        str(checkpoint / "actor"), del_local_after_load=False
    )


def test_disabled_resume_does_not_read_a_marker_or_load_state(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("")
    trainer = _trainer(tmp_path, "disable", tmp_path / "global_step_100")

    assert trainer._load_checkpoint() == 0
    assert trainer.global_steps == 0
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.train_dataloader.load_state_dict.assert_not_called()
