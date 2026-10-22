"""Unit tests of cerno_pi.decide that need neither a model nor libpinfer.

The tokenizer-dependent tests run when CERNO_TOKENIZER points at Cerno's tokenizer.json
(apertixai/cerno-4b); they check the prompt layout against strings rendered by hand.
"""
from __future__ import annotations

import math
import os
import zlib
from pathlib import Path

import numpy as np
import pytest

from cerno_pi.decide import (
    NEUTRAL_NONE,
    Decider,
    annotate_indices,
    build_row,
    common_prefix,
    format_answer,
    label_table,
    neutralize_options,
    readout,
    render_question,
    render_state,
)


def test_render_question_types() -> None:
    q = render_question({"type": "choice", "instructions": "Kind?",
                         "criteria": {"a": "first", "b": None, "c": ""}})
    assert q.options == ("a: first", "b", "c") and q.names == ("a", "b", "c")
    s = render_question({"type": "score", "instructions": "Level?", "criteria": ["low", "high"]})
    assert s.options == ("0: low", "1: high") and s.legend == ("low", "high")
    n = render_question({"type": "noul", "instructions": "Yes?",
                         "criteria": {"true": "it is", "false": None}})
    assert n.type == "noul" and n.options == ("no", "yes: it is")
    with pytest.raises(ValueError):
        render_question({"type": "choice", "instructions": "x", "criteria": ["only one"]})


def test_neutralize_and_state() -> None:
    assert neutralize_options(["A", "None of the above", "n/a", "none"]) == \
        ["A", NEUTRAL_NONE, NEUTRAL_NONE, NEUTRAL_NONE]
    assert render_state("text") == "text"
    arr = annotate_indices({"x": list(range(8)), "y": [1, 2]})
    assert arr["x"][3] == {"_index": 3, "value": 3} and arr["y"] == [1, 2]
    assert render_state({"a": [1, 2]}) == '{"a": [1, 2]}'


def test_common_prefix_keeps_a_suffix() -> None:
    assert common_prefix([[1, 2, 3, 4], [1, 2, 3, 5]]) == 3
    assert common_prefix([[1, 2], [1, 2, 3]]) == 1          # at most len(shortest) - 1
    assert common_prefix([[7, 8]]) == 1


def test_readout_equals_llama_formula() -> None:
    """softmax(l / T) == renormalised p^(1/T) of the full-vocabulary softmax (engine_llama)."""
    rng = np.random.default_rng(0)
    full = rng.normal(0, 3, 1000).astype(np.float32)
    lab = [3, 17, 400, 999]
    T = 1.534
    lse = math.log(sum(math.exp(float(x)) for x in full))
    p = [math.exp(float(full[i]) - lse) ** (1.0 / T) for i in lab]
    ref = [x / sum(p) for x in p]
    got = readout(full[lab], T)
    assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-12
    masked = readout(full[lab], T, np.array([True, False, True, True]))
    assert masked[1] == 0.0 and abs(sum(masked) - 1) < 1e-12


def test_format_answer_shapes() -> None:
    q = render_question({"type": "score", "instructions": "L?", "criteria": ["a", "b", "c"]})
    a = format_answer(q, [0.2, 0.5, 0.3])
    assert a["score"] == 1.1 and a["probabilities"] == {"0": 0.2, "1": 0.5, "2": 0.3}
    n = render_question({"type": "noul", "instructions": "Y?"})
    assert format_answer(n, [0.25, 0.75]) == {"type": "noul", "noul": 0.75}


TOK = os.environ.get("CERNO_TOKENIZER")


@pytest.mark.skipif(not TOK, reason="set CERNO_TOKENIZER=<cerno-4b>/tokenizer.json")
def test_row_layout_matches_hand_rendering() -> None:
    from pinfer import Tokenizer

    tok = Tokenizer.from_path(TOK)
    labels = label_table(tok)
    assert labels.names[:3] == ("A", "B", "C") and len(labels.ids) == 255
    ids = build_row(tok, labels, "hello", "Is it?", ["no", "yes"])
    text = "Context:\nhello\n\nQuestion: Is it?\nOptions:\n(A) no\n(B) yes\nAnswer: ("
    assert ids == tok.encode(text, add_special_tokens=False)
    wide = [f"opt{i}" for i in range(12)]
    ids = build_row(tok, labels, "s", "Which?", wide)
    assert tok.decode(ids).endswith("\n(L) opt11\nAnswer: (")


# ---------------------------------------------------------------- shared-prefix restore paths
class _CharTok:
    """Character-level stand-in for the tokenizer: one id per character, plus a single id
    for every label string (A..Z, AA..ZZ) so that label_table finds 255 labels."""

    BASE = 0x110000   # above every code point

    def __init__(self) -> None:
        up = [chr(65 + i) for i in range(26)]
        names = up + [a + b for a in up for b in up]
        self.ids = {n: self.BASE + i for i, n in enumerate(names)}
        self.names = {v: k for k, v in self.ids.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.ids[text]] if text in self.ids else [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.names.get(i) or chr(i) for i in ids)


class _FakeModel:
    """Duck-typed pinfer.Model: the 'logits' of a label are a hash of the whole context, so a
    restore that leaves anything else in context changes every probability."""

    def __init__(self, checkpoint: bool) -> None:
        self.tokenizer = _CharTok()
        self.path = Path("no-such-dir") / "fake.pq"
        self.max_ctx = 4096
        self.supports_state = True
        self.supports_checkpoint = checkpoint
        self.ctx: list[int] = []
        self.ck: list[int] | None = None
        self.calls: list[str] = []

    def reset(self) -> None:
        self.ctx, self.ck = [], None

    def eval_select(self, toks: list[int], ids: list[int]) -> np.ndarray:
        self.ctx += list(toks)
        h = zlib.crc32(np.asarray(self.ctx, dtype=np.int64).tobytes())
        return np.array([((h ^ (i * 2654435761)) % 1000) / 100.0 for i in ids],
                        dtype=np.float32)

    def save_state(self) -> bytes:
        self.calls.append("save")
        return np.asarray(self.ctx, dtype=np.int64).tobytes()

    def load_state(self, data: bytes) -> None:
        self.calls.append("load")
        self.ctx, self.ck = np.frombuffer(data, dtype=np.int64).tolist(), None

    def checkpoint(self) -> None:
        self.calls.append("checkpoint")
        self.ck = list(self.ctx)

    def rewind(self) -> None:
        self.calls.append("rewind")
        assert self.ck is not None
        self.ctx = list(self.ck)


QUESTIONS = {
    "kind": {"type": "choice", "instructions": "Which kind?", "criteria": ["a", "b", "c"]},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "level": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]},
}


def test_checkpoint_and_state_paths_agree_exactly() -> None:
    """The in-engine checkpoint, the state blob and no reuse score the same rows."""
    runs = {}
    for name, kw, ck in (("checkpoint", {}, True), ("state", {"checkpoint": False}, True),
                         ("fallback", {}, False), ("none", {"reuse_state": False}, True)):
        fm = _FakeModel(checkpoint=ck)
        res = Decider(fm, **kw).decide("the printer is on fire", QUESTIONS)  # type: ignore[arg-type]
        runs[name] = (res, fm.calls)
    assert runs["checkpoint"][1] == ["checkpoint", "rewind", "rewind"]
    assert runs["state"][1] == runs["fallback"][1] == ["save", "load", "load"]
    assert runs["none"][1] == []
    assert runs["checkpoint"][0].snapshot == "checkpoint"
    assert runs["state"][0].snapshot == runs["fallback"][0].snapshot == "state"
    assert runs["none"][0].snapshot == ""
    ref = runs["none"][0].probabilities
    for name in ("checkpoint", "state", "fallback"):
        assert runs[name][0].probabilities == ref, name
        assert runs[name][0].prefix_tokens > 0


def test_checkpoint_true_needs_the_api() -> None:
    with pytest.raises(NotImplementedError):
        Decider(_FakeModel(checkpoint=False), checkpoint=True)  # type: ignore[arg-type]
