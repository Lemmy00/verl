import math

import pytest
import torch

from verl.workers.actor.dp_actor import _capture_gradient_probe, _finish_gradient_probe
from verl.workers.config.actor import SftReplayConfig


def test_sft_replay_gradient_probe_recovers_exact_geometry_when_all_elements_are_sampled():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([0.0, 0.0])

    probe = _capture_gradient_probe([first, second], max_samples=4)
    # Replay gradient [4, -3, 0, 0] is orthogonal to the existing [3, 4, 0, 0].
    first.grad.add_(torch.tensor([4.0, -3.0]))
    metrics = _finish_gradient_probe(probe)

    assert metrics["actor/non_replay_grad_norm_est"] == pytest.approx(5.0)
    assert metrics["actor/sft_replay_grad_norm_est"] == pytest.approx(5.0)
    assert metrics["actor/combined_grad_norm_est"] == pytest.approx(math.sqrt(50.0))
    assert metrics["actor/sft_replay_to_non_replay_grad_ratio"] == pytest.approx(1.0)
    assert metrics["actor/sft_replay_grad_cosine"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["actor/sft_replay_grad_probe_fraction"] == pytest.approx(1.0)


def test_sft_replay_gradient_probe_accounts_for_gradient_scaling():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([6.0, 8.0])
    probe = _capture_gradient_probe([parameter], max_samples=2)
    parameter.grad.add_(torch.tensor([0.0, 10.0]))

    metrics = _finish_gradient_probe(probe, gradient_scale=2.0)

    assert metrics["actor/non_replay_grad_norm_est"] == pytest.approx(5.0)
    assert metrics["actor/sft_replay_grad_norm_est"] == pytest.approx(5.0)


def test_sft_replay_gradient_probe_can_be_disabled_or_have_no_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))

    assert _capture_gradient_probe([parameter], max_samples=2) is None
    parameter.grad = torch.ones(2)
    assert _capture_gradient_probe([parameter], max_samples=0) is None


def test_sft_replay_gradient_probe_rejects_negative_sample_budget():
    with pytest.raises(ValueError, match="grad_probe_samples must be non-negative"):
        SftReplayConfig(grad_probe_samples=-1)
