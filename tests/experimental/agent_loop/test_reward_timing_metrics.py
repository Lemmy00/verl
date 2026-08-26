from types import SimpleNamespace

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopManager


def test_streamed_reward_timing_exposes_true_critical_path():
    manager = AgentLoopManager.__new__(AgentLoopManager)
    metrics = [
        [
            {
                "generate_sequences": 10.0,
                "tool_calls": 0.0,
                "reward_score_s": 2.0,
                "num_preempted": 0,
            },
            {
                "generate_sequences": 8.0,
                "tool_calls": 0.0,
                "reward_score_s": 7.0,
                "num_preempted": 0,
            },
        ]
    ]
    output = SimpleNamespace(
        batch={
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0],
                ]
            ),
            "prompts": torch.ones((2, 2), dtype=torch.long),
        }
    )

    timing = manager._performance_metrics(metrics, output)

    assert timing["agent_loop/generate_sequences/max"] == 10.0
    assert timing["agent_loop/reward_score_s/max"] == 7.0
    assert timing["agent_loop/sample_completion/max"] == 15.0
    assert timing["agent_loop/lean_tail_after_generation_max"] == 5.0
    assert timing["agent_loop/critical_path/generate_sequences"] == 8.0
    assert timing["agent_loop/critical_path/reward_score_s"] == 7.0
    assert timing["agent_loop/critical_path/response_length"] == 3
