# Copyright 2026 Individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Retain complete local checkpoints across training process restarts."""

import datetime
import functools
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path


@functools.lru_cache(maxsize=1)
def selector():
    # The GRPO launcher supplies the same archive validator used for resume.
    path = Path(os.environ["GRPO_CHECKPOINT_SELECTOR_PATH"])
    spec = importlib.util.spec_from_file_location("recovery_checkpoint_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prune_checkpoints(root, keep, world_size=8):
    """Prune this run after a complete save, preserving newer partial saves."""
    root = Path(root)
    if keep < 2:
        raise ValueError("Cross-restart checkpoint retention requires keep >= 2")
    candidates = []
    for path in root.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match and path.is_dir() and not path.is_symlink() and path.resolve().parent == root.resolve():
            candidates.append((int(match[1]), path))
    complete = sorted(
        (step, path) for step, path in candidates if selector().incomplete_reason(path, world_size) is None
    )
    retained = complete[-keep:]
    removable = complete[:-keep]
    # The FSDP manager removes actor/ but can leave data.pt behind. Only clean
    # older remnants once a full window of complete checkpoints is available.
    if len(retained) == keep:
        complete_paths = {path for _, path in complete}
        removable += [(step, path) for step, path in candidates if path not in complete_paths and step < retained[0][0]]
    removed, errors = [], []
    for _, path in sorted(removable):
        try:
            shutil.rmtree(path)
            removed.append(str(path))
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})
    report = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "retained": [str(path) for _, path in retained],
        "removed": removed,
        "errors": errors,
    }
    if removed or errors:
        with (root / "checkpoint_retention.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print("Checkpoint retention: " + json.dumps(report), flush=True)
    return report


def after_checkpoint_save(config):
    if not config.trainer.get("resume_checkpoint_retention", False):
        return None
    return prune_checkpoints(
        config.trainer.default_local_dir,
        int(config.trainer.max_actor_ckpt_to_keep),
        int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes),
    )
