"""Unit tests of cerno_pi.decide that need neither a model nor libpinfer.

The tokenizer-dependent tests run when CERNO_TOKENIZER points at Cerno's tokenizer.json
(apertixai/cerno-4b); they check the prompt layout against strings rendered by hand.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from cerno_pi.decide import (
    NEUTRAL_NONE,
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
