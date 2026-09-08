# Copyright 2026 Individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Feedback masks must refer to sampled tokens, including noncanonical byte streams --
and, since 2026-09-08, only annotations INSIDE a proof body may be masked at all.

Position rule (collapse forensics): the malformed retry header "Here is /- <feedback>"
was fully PG-masked while the healthy header stayed exposed -- a one-way credit ratchet
at the exact broken transition. Masking now requires the span to sit wholly inside the
proof body (after ':= by') of a declaration inside a CLOSED ```lean4 fence; everywhere
else feedback-shaped text is a wrong ACTION and keeps its policy gradient. Boundary
rule: a token straddling a span edge carries action content too and stays eligible
(containment, not overlap).

The alignment fixtures therefore live inside a legitimate proof body; position-rule
pins get their own tests at the bottom.
"""

import random
import sys
from pathlib import Path

# The position-conditional mask locator soft-imports the reward repo's lexical
# helpers (lean_construction). Production launchers put the project src on
# PYTHONPATH; the test session must do the same BEFORE ray_trainer is imported,
# or the locator fails open and every masking assertion here trips.
_PROJECT_SRC = Path("/home/milikic/rl_theorem_provers/src")
if _PROJECT_SRC.is_dir() and str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _utf8_replacement_byte_boundaries

BODY_OPEN = b"```lean4\ntheorem t : True := by\n"
PREFIX = BODY_OPEN + b"proof\n"
# The genuine block-comment annotation form: round-2 Q1 demoted bare tags in code
# position (they are not Lean comments), so the shared alignment fixture must be a
# real comment for masking to engage at all.
FEEDBACK = b"/- <feedback>\nerror\n</feedback> -/\n"
SUFFIX = b"retry"
CLOSE = b"\n```"


@pytest.fixture(scope="module")
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.normalizer = normalizers.NFC()
    backend.train_from_iterator(
        [(PREFIX + FEEDBACK + SUFFIX).decode(), "é θ 日本語", "/- <feedback>\nerror\n</feedback> -/\n"],
        trainers.BpeTrainer(
            vocab_size=640,
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


def mask(tokenizer, ids, target="theorem t : True := by"):
    # Identity is a masking PREREQUISITE (round-2 Q2): the helper defaults to the
    # fixtures' declaration so alignment tests exercise masking; pass target=None to
    # test the identity-missing path (PG retained, its own metric).
    # Include padding, which must remain masked without being decoded as response.
    responses = torch.tensor([ids + [tokenizer.pad_token_id] * 3])
    response_mask = torch.tensor([[1] * len(ids) + [0] * 3])
    batch = DataProto.from_dict({"responses": responses, "response_mask": response_mask})
    if target is not None:
        import numpy as np
        batch.non_tensor_batch["theorem_statement_target"] = np.array([target], dtype=object)
    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = tokenizer
    metrics = {}
    result = trainer._proof_action_response_mask(batch, metrics)
    assert torch.equal(batch.batch["response_mask"], response_mask)
    assert result[0, len(ids) :].tolist() == [0, 0, 0]
    return result[0, : len(ids)].tolist(), metrics


def test_noncanonical_sampled_tokens_mask_feedback_and_preserve_proof(tokenizer):
    raw = PREFIX + FEEDBACK + SUFFIX + CLOSE
    ids = byte_ids(tokenizer, raw)
    assert len(tokenizer(raw.decode())["input_ids"]) < len(ids)
    result, metrics = mask(tokenizer, ids)
    assert result == [1] * len(PREFIX) + [0] * len(FEEDBACK) + [1] * (len(SUFFIX) + len(CLOSE))
    assert metrics["feedback/generated_feedback_tokenizer_mismatch_rows"] == 1
    assert metrics["feedback/generated_feedback_byte_alignment_rows"] == 1
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 0


def test_boundary_straddling_merged_token_stays_eligible(tokenizer):
    # Training merges the entire body+feedback+retry string into one token. Under the
    # OLD overlap rule that token was masked whole, silently deleting the proof and
    # retry action from PG. Under the containment rule it stays ELIGIBLE: a token
    # carrying action content must never be removed because feedback bytes ride in it.
    encoded = tokenizer((PREFIX + FEEDBACK + SUFFIX).decode())["input_ids"]
    assert len(encoded) == 1
    ids = [tokenizer.eos_token_id, *encoded, *byte_ids(tokenizer, CLOSE), tokenizer.eos_token_id]
    result, metrics = mask(tokenizer, ids)
    assert result == [1] * len(ids)
    assert metrics["feedback/generated_feedback_masked_tokens"] == 0.0


def test_hidden_special_inside_feedback_and_added_literal_token(tokenizer):
    added_id = tokenizer.convert_tokens_to_ids("<think>")
    ids = [added_id, *byte_ids(tokenizer, PREFIX + FEEDBACK[:12])]
    ids += [tokenizer.eos_token_id]
    ids += byte_ids(tokenizer, FEEDBACK[12:] + SUFFIX + CLOSE) + [tokenizer.eos_token_id]
    result, metrics = mask(tokenizer, ids)
    assert result == (
        [1] * (1 + len(PREFIX))
        + [0] * 12
        + [1]
        + [0] * (len(FEEDBACK) - 12)
        + [1] * (len(SUFFIX) + len(CLOSE) + 1)
    )
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        "θ 日本語 é".encode(),
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
    feedback = b"/- <feedback>\n" + extra + b"\n</feedback> -/\n"
    raw = prefix + feedback + SUFFIX + extra + CLOSE
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    assert result == [1] * len(prefix) + [0] * len(feedback) + [1] * (len(SUFFIX) + len(extra) + len(CLOSE))
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 0


def test_multiple_feedback_formats(tokenizer):
    first = b"/- <feedback>\nerror\n</feedback> -/\n"
    second = b"-- <feedback>\n-- error\n-- </feedback>\n"
    raw = PREFIX + first + SUFFIX + b"\n" + second + SUFFIX + CLOSE
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    assert result == (
        [1] * len(PREFIX)
        + [0] * len(first)
        + [1] * (len(SUFFIX) + 1)
        + [0] * len(second)
        + [1] * (len(SUFFIX) + len(CLOSE))
    )


def test_equal_length_reencoding_with_different_ids_is_not_trusted(tokenizer):
    class MisleadingOffsets:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, text, **kwargs):
            ids = byte_ids(tokenizer, text.encode())
            # Same token count but different identity: accepting these deliberately
            # wrong offsets would suppress proof and leave all feedback under PPO.
            return {"input_ids": list(reversed(ids)), "offset_mapping": [(len(PREFIX), len(PREFIX) + 1)] * len(ids)}

    raw = PREFIX + FEEDBACK + SUFFIX + CLOSE
    result, metrics = mask(MisleadingOffsets(), byte_ids(tokenizer, raw))
    assert result == [1] * len(PREFIX) + [0] * len(FEEDBACK) + [1] * (len(SUFFIX) + len(CLOSE))
    assert metrics["feedback/generated_feedback_byte_alignment_rows"] == 1


def test_unverifiable_decoder_retains_mask_and_reports_failure(tokenizer, caplog):
    class DifferentDecoder:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, text, **kwargs):
            raise NotImplementedError("offsets unavailable")

        def decode(self, ids, **kwargs):
            return "extra text " + tokenizer.decode(ids, **kwargs)

    ids = byte_ids(tokenizer, PREFIX + FEEDBACK + SUFFIX + CLOSE)
    result, metrics = mask(DifferentDecoder(), ids)
    assert result == [1] * len(ids)
    assert metrics["feedback/generated_feedback_alignment_failed_rows"] == 1
    assert "retaining their original PPO masks" in caplog.text


def test_no_feedback_leaves_response_unchanged(tokenizer):
    ids = byte_ids(tokenizer, PREFIX + SUFFIX + CLOSE)
    result, metrics = mask(tokenizer, ids)
    assert result == [1] * len(ids)
    assert not metrics


# ---------------------------------------------------------------------------------
# Position-rule pins (2026-09-08). Every case below was PG-masked under the old
# recognizer; each is a wrong ACTION and must now stay eligible, with the drop counted.
# ---------------------------------------------------------------------------------


def _all_eligible(tokenizer, raw):
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    return result == [1] * len(result), metrics


def test_malformed_retry_header_feedback_stays_eligible(tokenizer):
    """The collapse's signature: 'Here is /- <feedback>...' in prose position."""
    raw = PREFIX + b"done\n" + CLOSE + b"\n\nHere is /- <feedback>\nerror\n</feedback> -/\n"
    ok, metrics = _all_eligible(tokenizer, raw)
    assert ok
    assert metrics["feedback/feedback_spans_outside_proof_rows"] == 1
    assert metrics["feedback/feedback_spans_outside_proof_count"] == 1.0


def test_feedback_outside_any_fence_stays_eligible(tokenizer):
    ok, metrics = _all_eligible(tokenizer, b"prose " + FEEDBACK + b"more prose")
    assert ok
    assert metrics["feedback/feedback_spans_outside_proof_rows"] == 1


def test_feedback_in_statement_region_before_by_stays_eligible(tokenizer):
    raw = b"```lean4\ntheorem t : " + FEEDBACK + b"True := by\nproof\n" + CLOSE
    ok, metrics = _all_eligible(tokenizer, raw)
    assert ok
    assert metrics["feedback/feedback_spans_outside_proof_count"] == 1.0


def test_feedback_in_misnested_fence_is_ambiguous_not_malformed(tokenizer):
    """Phase-2 revision: a complete comment in an UNCLOSED FINAL fence is maskable
    (the budget ended the fence, not the model) -- that case is pinned above. The
    fail-open case that remains is a mis-nested fence: an opener never closed before
    the NEXT opener. Its spans stay eligible and are counted as AMBIGUOUS, never as
    malformed headers -- the review required these diagnostics to stay separate."""
    raw = (
        b"```lean4\ntheorem t : True := by\n" + FEEDBACK  # opener never closed...
        + b"```lean4\ntheorem t : True := by\n  trivial\n```"  # ...before the next
    )
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    start = raw.index(FEEDBACK)
    assert all(v == 1 for v in result[start : start + len(FEEDBACK)])
    assert metrics["feedback/feedback_spans_ambiguous_rows"] == 1
    assert metrics.get("feedback/feedback_spans_outside_proof_rows", 0) == 0


def test_in_body_and_out_of_body_spans_split_correctly(tokenizer):
    """One row, both kinds: only the in-body annotation is masked."""
    outside = b"\n\nHere is /- <feedback>\nboom\n</feedback> -/\n"
    raw = PREFIX + FEEDBACK + SUFFIX + CLOSE + outside
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    expected = (
        [1] * len(PREFIX)
        + [0] * len(FEEDBACK)
        + [1] * (len(SUFFIX) + len(CLOSE) + len(outside))
    )
    assert result == expected
    assert metrics["feedback/feedback_spans_outside_proof_rows"] == 1


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


# ---------------------------------------------------------------------------------
# Phase-2 review pins (2026-09-08). P1a: the recognizer includes an annotation's
# trailing newline; a TERMINAL in-proof comment therefore ended one character past the
# regex body and lost its mask on 194-219 rows per healthy step. P1b: the regex
# locator hallucinated proof bodies from binder defaults, comments, strings, and
# helper declarations. The lexical locator must survive every counterexample.
# ---------------------------------------------------------------------------------

FB_BLOCK = b"/- <feedback>\n-- type: error, msg: example\n</feedback> -/\n"


def _span_masked(result, raw, needle):
    start = raw.index(needle)
    return all(v == 0 for v in result[start : start + len(needle)])


@pytest.mark.parametrize(
    "tail",
    [b"```", b"\n```", b"   \n```", b"\r\n```"],
    ids=["fence-immediate", "blank-line", "indent-blank", "crlf"],
)
def test_terminal_feedback_keeps_its_mask(tokenizer, tail):
    """P1a: a terminal annotation whose trailing newline touches the closing fence is
    the mask's core case -- semantic-end containment must accept it."""
    raw = b"```lean4\ntheorem t : True := by\n  trivial\n" + FB_BLOCK + tail
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    core = FB_BLOCK.rstrip(b"\n")
    assert _span_masked(result, raw, core)
    assert metrics.get("feedback/feedback_spans_outside_proof_rows", 0) == 0


def test_binder_default_does_not_open_a_proof_body(tokenizer):
    """P1b: '(h : True := by trivial)' is a binder default, not the proof opening."""
    raw = (
        b"```lean4\ntheorem t (h : True := by trivial)\n"
        + FB_BLOCK
        + b": True := by\n  exact h\n```"
    )
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))
    assert metrics["feedback/feedback_spans_outside_proof_rows"] == 1


def test_decl_text_inside_comment_is_not_a_body(tokenizer):
    raw = (
        b"```lean4\n-- note: theorem fake : True := by trivial\n"
        + FB_BLOCK
        + b"def x := 1\n```"
    )
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_feedback_marker_inside_string_literal_is_code(tokenizer):
    raw = (
        b'```lean4\ntheorem t : True := by\n  have s := "/- <feedback>\nx\n</feedback> -/"\n  trivial\n```'
    )
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    assert result == [1] * len(result)


def test_second_declaration_statement_is_not_first_body(tokenizer):
    raw = (
        b"```lean4\ntheorem a : True := by\n  trivial\n"
        b"theorem b (x : Nat)\n" + FB_BLOCK + b": True := by\n  trivial\n```"
    )
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_target_identity_excludes_helper_declaration_bodies(tokenizer):
    """With the requested theorem known, a helper lemma's body does not mask."""
    raw = (
        b"```lean4\nlemma helper : True := by\n  " + FB_BLOCK + b"  trivial\n"
        b"theorem target : True := by\n  " + FB_BLOCK + b"  trivial\n```"
    )
    result, _ = mask(
        tokenizer, byte_ids(tokenizer, raw), target="theorem target : True := by"
    )
    first = raw.index(FB_BLOCK.rstrip(b"\n"))
    second = raw.index(FB_BLOCK.rstrip(b"\n"), first + 1)
    core = FB_BLOCK.rstrip(b"\n")
    assert any(v == 1 for v in result[first : first + len(core)])  # helper: eligible
    assert all(v == 0 for v in result[second : second + len(core)])  # target: masked


def test_complete_comment_in_unclosed_final_fence_is_masked(tokenizer):
    """A budget-truncated proof keeps its legitimate annotations masked."""
    raw = b"```lean4\ntheorem t : True := by\n  " + FB_BLOCK + b"  exact"
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    assert _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_missing_identity_retains_pg_and_reports_prerequisite(tokenizer):
    raw = PREFIX + FEEDBACK + SUFFIX + CLOSE
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw), target=None)
    assert result == [1] * len(result)
    assert metrics["feedback/feedback_mask_identity_missing_rows"] == 1
    assert metrics.get("feedback/feedback_spans_outside_proof_rows", 0) == 0


def test_line_comment_sandwich_keeps_the_tactic_eligible(tokenizer):
    """Round-2 P1 counterexample 1 (compiles under Lean 4.15): an executable tactic
    between '--' feedback lines. The recognizer span bridges it; full comment
    coverage must refuse to mask the code line."""
    raw = (
        b"```lean4\ntheorem t : True := by\n"
        b"-- <feedback>\n  exact True.intro\n-- </feedback>\n```"
    )
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    tactic = b"exact True.intro"
    start = raw.index(tactic)
    assert all(v == 1 for v in result[start : start + len(tactic)])


def test_split_block_comments_keep_the_tactic_eligible(tokenizer):
    """Round-2 P1 counterexample 2: two separate complete block comments whose
    combined recognizer span bridges live code."""
    raw = (
        b"```lean4\ntheorem t : True := by\n"
        b"/- <feedback>\n-/\n  exact True.intro\n/-\n</feedback> -/\n```"
    )
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    tactic = b"exact True.intro"
    start = raw.index(tactic)
    assert all(v == 1 for v in result[start : start + len(tactic)])


def test_bare_tags_in_code_position_do_not_mask_a_tactic(tokenizer):
    """Round-2 Q1 ruling: '<feedback>' is not a Lean comment; wrapping a tactic in
    bare tags must not strip its policy gradient."""
    raw = (
        b"```lean4\ntheorem t : True := by\n"
        b"<feedback>\n  exact True.intro\n</feedback>\n```"
    )
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    assert result == [1] * len(result)


def test_bare_tags_inside_a_genuine_comment_still_mask(tokenizer):
    raw = (
        b"```lean4\ntheorem t : True := by\n  trivial\n"
        b"/- <feedback>\nerror text\n</feedback> -/\n```"
    )
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw))
    inner = b"/- <feedback>\nerror text\n</feedback> -/"
    start = raw.index(inner)
    assert all(v == 0 for v in result[start : start + len(inner)])


@pytest.mark.parametrize(
    "later",
    [
        b"def helper\n" + FB_BLOCK + b": Nat := 0\n",
        b"instance\n" + FB_BLOCK + b": Inhabited Nat := \xe2\x9f\xa80\xe2\x9f\xa9\n",
        b"def helper : Nat :=\n" + FB_BLOCK + b"  0\n",
    ],
    ids=["def-statement", "instance-statement", "def-body"],
)
def test_later_non_theorem_declarations_end_the_body(tokenizer, later):
    """Round-2 P1: def/instance after the target must terminate its mask region."""
    raw = (
        b"```lean4\ntheorem t : True := by\n  trivial\n" + later + b"```"
    )
    result, metrics = mask(tokenizer, byte_ids(tokenizer, raw))
    core = FB_BLOCK.rstrip(b"\n")
    start = raw.index(core)
    assert any(v == 1 for v in result[start : start + len(core)])
    assert metrics["feedback/feedback_spans_outside_proof_rows"] == 1


# ---------------------------------------------------------------------------
# Round-3 P1: the boundary recognizer must be structural. A bare keyword regex
# missed modifier-prefixed and indented declarations (six feedback comments in
# LATER declarations stayed masked) and truncated a proof at a `set_option ... in`
# TACTIC (losing a legitimate terminal annotation). All fixtures compile under
# Lean 4.15 (phase2_round3_boundary_recheck.json).

TARGET = "theorem target : True := by"


def _boundary_mask(tokenizer, raw):
    result, _ = mask(tokenizer, byte_ids(tokenizer, raw), target=TARGET)
    return result, raw


def test_private_def_statement_is_not_target_body(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"private def helper\n" + FB_BLOCK + b": Nat := 0\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_noncomputable_def_body_is_not_target_body(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"noncomputable def helper : Nat := by\n" + FB_BLOCK + b"  exact 0\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_local_instance_statement_is_not_target_body(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"local instance\n" + FB_BLOCK + b": Inhabited Nat := \xe2\x9f\xa80\xe2\x9f\xa9\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_private_theorem_statement_is_not_target_body(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"private theorem helper\n" + FB_BLOCK + b": True := by trivial\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_indented_def_in_namespace_ends_target_body(tokenizer):
    raw = (
        b"```lean4\nnamespace N\n  theorem target : True := by\n    trivial\n"
        b"  def helper\n" + FB_BLOCK + b"  : Nat := 0\nend N\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_indented_attributed_def_ends_target_body(tokenizer):
    raw = (
        b"```lean4\nnamespace N\n  theorem target : True := by\n    trivial\n"
        b"  @[simp] def helper\n" + FB_BLOCK + b"  : Nat := 0\nend N\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_unindented_attributed_def_ends_target_body(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"@[simp] def helper\n" + FB_BLOCK + b": Nat := 0\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_set_option_in_tactic_keeps_terminal_feedback_masked(tokenizer):
    """A one-space `set_option ... in` TACTIC must not end the proof: the terminal
    annotation after it is a legitimate in-proof comment and must stay masked."""
    raw = (
        b"```lean4\ntheorem target : True := by\n"
        b" set_option maxRecDepth 1000 in\n  exact True.intro\n" + FB_BLOCK + b"```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_open_in_tactic_keeps_terminal_feedback_masked(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n"
        b" open Nat in\n  exact True.intro\n" + FB_BLOCK + b"```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_one_space_induction_control_masks_terminal_feedback(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n"
        b" induction (0:Nat) with\n | zero => exact True.intro\n"
        b" | succ n ih => exact True.intro\n" + FB_BLOCK + b"```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_set_option_chain_to_declaration_is_a_boundary(tokenizer):
    """Command-position `set_option ... in` prefixing a declaration DOES end the
    target's body -- the chain's continuation decides tactic vs command."""
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"set_option pp.all true in\ndef helper\n" + FB_BLOCK + b": Nat := 0\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


def test_standalone_set_option_command_is_a_boundary(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"set_option pp.all true\n" + FB_BLOCK + b"```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert not _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


@pytest.mark.parametrize("prefix", [b"private\n", b"noncomputable\r\n", b"@[simp]\nprivate\n"])
def test_multiline_modifier_starts_the_next_declaration(tokenizer, prefix):
    """Compiled round-four cases: the comment is after helper's modifier, not in target."""
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        + prefix + FB_BLOCK + b"def helper : Nat := 0\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert result == [1] * len(raw)


@pytest.mark.parametrize("header", [
    b"set_option maxRecDepth 1000\n  in",
    b"open Nat\n  in",
    b"set_option\n    maxRecDepth\n    1000\n  in",
    b"open\n    Nat\n  in",
    b"open Nat\n  /- a comment between the arguments and in -/\n  in",
])
def test_multiline_in_tactic_keeps_terminal_feedback_masked(tokenizer, header):
    raw = (
        b"```lean4\ntheorem target : True := by\n  " + header
        + b"\n    trivial\n" + FB_BLOCK + b"```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert _span_masked(result, raw, FB_BLOCK.rstrip(b"\n"))


@pytest.mark.parametrize("header", [b"set_option maxRecDepth 1000", b"open Nat"])
def test_multiline_in_command_chain_ends_target_body(tokenizer, header):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n" + header
        + b"\nin\n" + FB_BLOCK + b"def helper : Nat := 0\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert result == [1] * len(raw)


@pytest.mark.parametrize("command", [b"#check", b"#eval"])
def test_hash_command_feedback_does_not_belong_to_target(tokenizer, command):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n" + command
        + b"\n" + FB_BLOCK + b"(0 : Nat)\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert result == [1] * len(raw)


def test_standalone_open_does_not_borrow_in_from_a_later_proof(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\nopen Nat\n"
        + FB_BLOCK + b"example : True := by\n  set_option maxRecDepth 1000 in\n    trivial\n```"
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert result == [1] * len(raw)


def test_incomplete_command_chain_retains_pg(tokenizer):
    raw = (
        b"```lean4\ntheorem target : True := by\n  trivial\n"
        b"set_option maxRecDepth 1000\nin\n" + FB_BLOCK
    )
    result, raw = _boundary_mask(tokenizer, raw)
    assert result == [1] * len(raw)
