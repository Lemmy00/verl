# Copyright 2026 Individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Feedback masks must refer to sampled tokens, including noncanonical byte streams."""

import random

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _utf8_replacement_byte_boundaries

PREFIX = b"proof\n"
FEEDBACK = b"<feedback>\nerror\n</feedback>\n"
SUFFIX = b"retry"


@pytest.fixture(scope="module")
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.normalizer = normalizers.NFC()
    backend.train_from_iterator(
        [(PREFIX + FEEDBACK + SUFFIX).decode(), "é θ 日本語", "/- <feedback>\nerror\n</feedback> -/\n"],
        trainers.BpeTrainer(
            vocab_size=360,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<|eos|>", "<|pad|>"],
            show_progress=False,
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        clean_up_tokenization_spaces=False,
    )
    tokenizer.add_tokens(["<think>"])
    return tokenizer


def byte_ids(tokenizer, raw):
    encoder = bytes_to_unicode()
    return [tokenizer.convert_tokens_to_ids(encoder[byte]) for byte in raw]


def mask(tokenizer, ids):
    # Include padding, which must remain masked without being decoded as response.
    responses = torch.tensor([ids + [tokenizer.pad_token_id] * 3])
    response_mask = torch.tensor([[1] * len(ids) + [0] * 3])
    batch = DataProto.from_dict({"responses": responses, "response_mask": response_mask})
    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = tokenizer
    metrics = {}
    result = trainer._proof_action_response_mask(batch, metrics)
    assert torch.equal(batch.batch["response_mask"], response_mask)
    assert result[0, len(ids) :].tolist() == [0, 0, 0]
    return result[0, : len(ids)].tolist(), metrics


def test_noncanonical_sampled_tokens_mask_feedback_and_preserve_proof(tokenizer):
    raw = PREFIX + FEEDBACK + SUFFIX
    ids = byte_ids(tokenizer, raw)
    assert len(tokenizer(raw.decode())["input_ids"]) < len(ids)
    result, metrics = mask(tokenizer, ids)
    assert result == [1] * len(PREFIX) + [0] * len(FEEDBACK) + [1] * len(SUFFIX)
    assert metrics["feedback/generated_feedback_tokenizer_mismatch_rows"] == 1
    assert metrics["feedback/generated_feedback_byte_alignment_rows"] == 1
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 0


def test_canonical_merged_boundary_token_and_hidden_specials(tokenizer):
    # Training merges the entire string into one token. It overlaps feedback and
    # must be excluded as a whole, even though it also contains proof characters.
    encoded = tokenizer((PREFIX + FEEDBACK + SUFFIX).decode())["input_ids"]
    assert len(encoded) == 1
    ids = [tokenizer.eos_token_id, *encoded, tokenizer.eos_token_id]
    result, metrics = mask(tokenizer, ids)
    assert result == [1, 0, 1]
    assert metrics["feedback/generated_feedback_tokenizer_mismatch_rows"] == 0
    assert metrics["feedback/generated_feedback_byte_alignment_rows"] == 0


def test_hidden_special_inside_feedback_and_added_literal_token(tokenizer):
    added_id = tokenizer.convert_tokens_to_ids("<think>")
    ids = [added_id, *byte_ids(tokenizer, PREFIX + FEEDBACK[:12])]
    ids += [tokenizer.eos_token_id]
    ids += byte_ids(tokenizer, FEEDBACK[12:] + SUFFIX) + [tokenizer.eos_token_id]
    result, metrics = mask(tokenizer, ids)
    assert result == [1] * (1 + len(PREFIX)) + [0] * 12 + [1] + [0] * (len(FEEDBACK) - 12) + [1] * (len(SUFFIX) + 1)
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        "θ 日本語 e\u0301".encode(),
        b"\xff",
        b"\xe2\x82",
        b"\xed\xa0",
        b"\xed\xa0\x80",
        b"\xed\xa0X",
        b"\xed\xa0\xc3\xa9",
    ],
)
def test_split_unicode_and_invalid_bytes_before_and_inside_feedback(tokenizer, extra):
    prefix = PREFIX + extra
    feedback = b"<feedback>\n" + extra + b"\n</feedback>\n"
    raw = prefix + feedback + SUFFIX + extra
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    assert result == [1] * len(prefix) + [0] * len(feedback) + [1] * (len(SUFFIX) + len(extra))
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 0


def test_multiple_feedback_formats(tokenizer):
    first = b"/- <feedback>\nerror\n</feedback> -/\n"
    second = b"-- <feedback>\n-- error\n-- </feedback>\n"
    raw = PREFIX + first + SUFFIX + second + SUFFIX
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    assert result == [1] * len(PREFIX) + [0] * len(first) + [1] * len(SUFFIX) + [0] * len(second) + [1] * len(SUFFIX)


def test_equal_length_reencoding_with_different_ids_is_not_trusted(tokenizer):
    class MisleadingOffsets:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, text, **kwargs):
            ids = byte_ids(tokenizer, text.encode())
            # Same token count but different identity: accepting these deliberately
            # wrong offsets would suppress proof and leave all feedback under PPO.
            return {"input_ids": list(reversed(ids)), "offset_mapping": [(len(PREFIX), len(PREFIX) + 1)] * len(ids)}

    raw = PREFIX + FEEDBACK + SUFFIX
    result, metrics = mask(MisleadingOffsets(), byte_ids(tokenizer, raw))
    assert result == [1] * len(PREFIX) + [0] * len(FEEDBACK) + [1] * len(SUFFIX)
    assert metrics["feedback/generated_feedback_byte_alignment_rows"] == 1


def test_unverifiable_decoder_retains_mask_and_reports_failure(tokenizer, caplog):
    class DifferentDecoder:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, text, **kwargs):
            raise NotImplementedError("offsets unavailable")

        def decode(self, ids, **kwargs):
            return "extra text " + tokenizer.decode(ids, **kwargs)

    ids = byte_ids(tokenizer, PREFIX + FEEDBACK + SUFFIX)
    result, metrics = mask(DifferentDecoder(), ids)
    assert result == [1] * len(ids)
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 1
    assert "retaining their original PPO masks" in caplog.text


def test_no_feedback_leaves_response_unchanged(tokenizer):
    ids = byte_ids(tokenizer, PREFIX + SUFFIX)
    result, metrics = mask(tokenizer, ids)
    assert result == [1] * len(ids)
    assert not metrics


def test_utf8_replacement_boundaries_recover_individual_characters():
    rng = random.Random(20260905)
    examples = [b"\xed\xa0", b"\xed\xa0\x80", b"\xed\xa0X", b"\xed\xa0\xc3\xa9"]
    examples += [rng.randbytes(rng.randrange(100)) for _ in range(1000)]
    for raw in examples:
        text = raw.decode("utf-8", errors="replace")
        boundaries = _utf8_replacement_byte_boundaries(raw)
        assert len(boundaries) == len(text) + 1
        assert boundaries[0] == 0 and boundaries[-1] == len(raw)
        assert [
            raw[start:end].decode("utf-8", errors="replace")
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ] == list(text)
