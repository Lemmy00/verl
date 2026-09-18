# Copyright 2026 Individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Retention must preserve usable recovery states and stay inside its run."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo import recovery_checkpoint as recovery


@pytest.fixture
def checkpoints(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()

    def incomplete_reason(path, world_size):
        return None if (path / "complete").is_file() else "interrupted save"

    monkeypatch.setattr(recovery, "selector", lambda: SimpleNamespace(incomplete_reason=incomplete_reason))

    def create(step, complete=True):
        path = root / f"global_step_{step}"
        path.mkdir()
        if complete:
            (path / "complete").touch()
        return path

    return root, create


def test_retention_survives_restarts_and_skips_newer_partial_saves(checkpoints):
    root, create = checkpoints
    saves = {step: create(step) for step in (10, 20, 30, 40)}
    partial = create(50, complete=False)
    first = recovery.prune_checkpoints(root, keep=3)
    assert first["removed"] == [str(saves[10])]
    assert partial.exists()
    # No in-memory history is carried to the next call, as after a restart.
    create(60)
    second = recovery.prune_checkpoints(root, keep=3)
    assert second["removed"] == [str(saves[20])]
    assert {p.name for p in root.glob("global_step_*")} == {
        "global_step_30",
        "global_step_40",
        "global_step_50",
        "global_step_60",
    }
    assert len((root / "checkpoint_retention.jsonl").read_text().splitlines()) == 2


def test_old_partial_save_removed_only_after_enough_complete_saves(checkpoints):
    root, create = checkpoints
    old_partial = create(5, complete=False)
    create(10)
    create(20)
    assert recovery.prune_checkpoints(root, keep=3)["removed"] == []
    assert old_partial.exists()
    create(30)
    assert recovery.prune_checkpoints(root, keep=3)["removed"] == [str(old_partial)]


def test_symlink_and_unrelated_files_are_never_pruned(checkpoints, tmp_path):
    root, create = checkpoints
    outside = tmp_path / "preserved_model"
    outside.mkdir()
    (outside / "complete").touch()
    link = root / "global_step_1"
    link.symlink_to(outside, target_is_directory=True)
    inputs = root / "inputs.parquet"
    inputs.write_bytes(b"keep input data")
    for step in (10, 20, 30, 40):
        create(step)
    recovery.prune_checkpoints(root, keep=3)
    assert link.is_symlink() and (outside / "complete").exists()
    assert inputs.read_bytes() == b"keep input data"


def test_disabled_hook_does_not_require_selector_or_checkpoint_directory(monkeypatch):
    monkeypatch.delenv("GRPO_CHECKPOINT_SELECTOR_PATH", raising=False)
    assert recovery.after_checkpoint_save(OmegaConf.create({"trainer": {}})) is None


def test_hook_uses_configured_root_retention_and_world_size(monkeypatch):
    calls = []
    monkeypatch.setattr(recovery, "prune_checkpoints", lambda *args: calls.append(args))
    config = OmegaConf.create(
        {
            "trainer": {
                "resume_checkpoint_retention": True,
                "default_local_dir": "/run/checkpoints",
                "max_actor_ckpt_to_keep": 3,
                "n_gpus_per_node": 4,
                "nnodes": 2,
            }
        }
    )
    recovery.after_checkpoint_save(config)
    assert calls == [("/run/checkpoints", 3, 8)]


def test_deletion_failure_is_reported_without_losing_newer_saves(checkpoints, monkeypatch):
    root, create = checkpoints
    saves = [create(step) for step in (10, 20, 30, 40)]

    def fail(path):
        raise OSError("read-only test checkpoint")

    monkeypatch.setattr(recovery.shutil, "rmtree", fail)
    report = recovery.prune_checkpoints(root, keep=3)
    assert len(report["errors"]) == 1
    assert all(path.exists() for path in saves)
    assert json.loads((root / "checkpoint_retention.jsonl").read_text())["errors"] == report["errors"]
