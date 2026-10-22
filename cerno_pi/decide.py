"""Typed decisions with calibrated probabilities from one forward pass (Cerno / decider models).

A decision model reads a *state* (text or JSON) and typed *questions* (``choice`` with named
options, ``noul`` yes/no, ``score`` with ordered levels) and returns a probability for every
option, read from the logits of the option-label tokens at one answer slot; no text is
generated. This module reproduces the scoring protocol of Cerno's llama.cpp engine
(``evals/decision_index/engine_llama.py``), which is also the vLLM engine's protocol:

* one row per question: ``"Context:\\n" + state`` then the question block
  ``"\\n\\nQuestion: ...\\nOptions:\\n(A) ...\\n(B) ...\\nAnswer: ("`` (decider's state-first
  layout; up to 10 options are lettered A..J and tokenized together with the state, more
  options get one label token each: A..Z, then two-letter tokens);
* the logits of the label tokens at the last position, ``p = softmax(logits / T)`` with the
  model's calibration temperature ``T`` (``decider_config.json``; Cerno 4B: 1.534). The
  reference reads llama-server's top-260 probabilities and renormalises ``p^(1/T)``: the same
  numbers, except that a label outside the top 260 counts as 0 there (``top_n=260`` emulates it);
* options that read like "none of the above" are shown to the model as "not listed here".

Every row starts with the same state tokens, so the engine prefills the common prefix once,
checkpoints it and scores each question from the checkpoint; the engine's batched and
token-by-token paths are bit-identical, so the result equals scoring every row from scratch.
The checkpoint is the engine's in-context one (:meth:`pinfer.Model.checkpoint` /
:meth:`pinfer.Model.rewind`: only the recurrent state is copied, the prefix's KV rows stay in
place) when the library has it, else a state blob (:meth:`pinfer.Model.save_state` /
:meth:`pinfer.Model.load_state`); both give the same probabilities bit for bit. Only the label
rows of the output head are computed (:meth:`pinfer.Model.eval_select`).

Portions are adapted from decider (https://github.com/Mapika/decider, commit 1c15a48f,
Apache-2.0) as vendored and modified in cerno-inference (https://github.com/apertixsk/
cerno-inference, commit 9941548d, Apache-2.0): the prompt layout and label table
(``decider/prompt.py``: ``build``, ``label_table``, ``JOINT_FIRST_PIECE``), typed questions and
answers (``decider/systemone.py``: ``render_state``, ``annotate_indices``, ``render_question``,
``format_answer``, ``certainty``), ``neutralize_options`` (``decider/infer.py``) and the readout
of ``engine_llama.py``. Changes: rewritten as typed functions for one-question rows (the
engines' ``plan_rows(..., isolated=False)`` path; isolated score levels are not supported), the
readout computed from engine logits, the shared-prefix snapshot.
"""
from __future__ import annotations

import json
import math
import string
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pinfer import errors

if TYPE_CHECKING:
    from pinfer import Model, Tokenizer

LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)          # <= 10 options: "(A) .. (J)" tokenized as one string
MAX_OPTIONS = 255              # > 10 options: one label token each (A..Z, AA, AB, ...)
MAX_CHOICE, MAX_LEVELS = 255, 10
ANNOTATE_MIN = 8
NEUTRAL_NONE = "not listed here"
DEFAULT_MAX_STATE_TOKENS = 16384   # engine_llama.py default
LLAMA_TOP_N = 260                  # engine_llama.py n_probs


# ---------------------------------------------------------------------- questions (systemone)
def _txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    """Write each element's position into long JSON arrays ({"_index": i, ...})."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)})
                    for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state: Any, index_arrays: bool = True) -> str:
    """Text as is; JSON values serialised (json.dumps defaults, long arrays indexed)."""
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state) if index_arrays else state, ensure_ascii=False)


@dataclass(frozen=True)
class Question:
    """A rendered question: model-facing text and options, and what the answer reports."""
    type: str                      # "choice" | "noul" | "score"
    text: str
    options: tuple[str, ...]
    names: tuple[Any, ...]
    legend: tuple[str, ...] | None = None


def render_question(spec: Mapping[str, Any]) -> Question:
    """``{"type", "instructions", "criteria"}`` -> :class:`Question` (decider's rendering)."""
    t = spec.get("type", "choice")
    ins = _txt(spec.get("instructions", spec.get("question", "")))
    crit = spec.get("criteria", spec.get("options"))
    if not ins:
        raise ValueError("question without instructions")
    legend = None
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_CHOICE:
            raise ValueError(f"choice criteria: a map of 2..{MAX_CHOICE} options")
        names: tuple[Any, ...] = tuple(crit)
        opts = [n if crit[n] in (None, "") else f"{n}: {_txt(crit[n])}" for n in names]
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in sorted(crit, key=float)]
        if not isinstance(crit, (list, tuple)) or not 2 <= len(crit) <= MAX_LEVELS:
            raise ValueError(f"score criteria: an ordered list of 2..{MAX_LEVELS} level descriptions")
        names = tuple(range(len(crit)))
        opts = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]
        legend = tuple(_txt(c) for c in crit)
    elif t in ("noul", "bool"):
        names = (False, True)
        c = crit or {}
        f, tr = c.get("false", c.get(False)), c.get("true", c.get(True))
        opts = ["no" if f in (None, "") else f"no: {_txt(f)}",
                "yes" if tr in (None, "") else f"yes: {_txt(tr)}"]
        t = "noul"
    else:
        raise ValueError(f"unknown question type {t!r}")
    return Question(t, ins, tuple(opts), names, legend)


def neutralize_options(options: Sequence[str]) -> list[str]:
    """Options that read like "none of the above" are shown as "not listed here"."""
    out = []
    for o in options:
        key = o.strip().lower()
        if key.startswith("none of the above") or key in ("none of the above", "none", "n/a",
                                                          "none of these"):
            out.append(NEUTRAL_NONE)
        else:
            out.append(o)
    return out


def certainty(p: Sequence[float]) -> float:
    """1 - normalised entropy."""
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(len(p))) if len(p) > 1 else 1.0


def format_answer(q: Question, p: Sequence[float], nd: int = 4) -> dict[str, Any]:
    """decider's answer shape (probabilities rounded to ``nd`` digits)."""
    ps = [float(x) for x in p[:len(q.options)]]
    s = sum(ps) or 1.0
    ps = [x / s for x in ps]
    j = max(range(len(ps)), key=ps.__getitem__)
    if q.type == "noul":
        return {"type": "noul", "noul": round(ps[1], nd)}
    if q.type == "choice":
        return {"type": "choice", "choice": q.names[j], "confidence": round(ps[j], nd),
                "certainty": round(certainty(ps), nd),
                "probabilities": {n: round(x, nd) for n, x in zip(q.names, ps)}}
    assert q.legend is not None
    return {"type": "score", "score": round(sum(i * x for i, x in enumerate(ps)), 2),
            "confidence": round(ps[j], nd), "certainty": round(certainty(ps), nd),
            "legend": {str(i): d for i, d in enumerate(q.legend)},
            "probabilities": {str(i): round(x, nd) for i, x in enumerate(ps)}}


# ---------------------------------------------------------------------- prompt (decider.prompt)
@dataclass(frozen=True)
class LabelTable:
    names: tuple[str, ...]         # MAX_OPTIONS label strings, A..J first
    ids: tuple[int, ...]           # their single token ids
    open_ids: tuple[int, ...]      # tokens of "\n("


def label_table(tok: Tokenizer) -> LabelTable:
    """A..Z then the two-letter upper-case strings that are single tokens, 255 in all."""
    upper = string.ascii_uppercase
    out: list[tuple[str, int]] = []
    for n in list(upper) + [a + b for a in upper for b in upper]:
        t = tok.encode(n, add_special_tokens=False)
        if len(t) == 1:
            out.append((n, t[0]))
        if len(out) == MAX_OPTIONS:
            break
    if len(out) != MAX_OPTIONS or len({i for _, i in out}) != MAX_OPTIONS:
        raise ValueError("tokenizer has fewer than 255 single-token labels")
    return LabelTable(tuple(n for n, _ in out), tuple(i for _, i in out),
                      tuple(tok.encode("\n(", add_special_tokens=False)))


def build_row(tok: Tokenizer, labels: LabelTable, state_text: str, question: str,
              options: Sequence[str], max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS
              ) -> list[int]:
    """Token ids of one state-first row with one question; the answer slot is the last token.
    Same ids as decider's ``build`` (one question, options kept in order, Cerno's
    JOINT_FIRST_PIECE)."""
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"2..{MAX_OPTIONS} options required")
    ctx_text = "Context:\n" + state_text
    ctx_ids = tok.encode(ctx_text, add_special_tokens=False)
    if len(ctx_ids) > max_state_tokens:        # truncate at the token level, back to text
        ctx_ids = ctx_ids[:max_state_tokens]
        ctx_text = tok.decode(ctx_ids)
    head = f"\n\nQuestion: {question}\nOptions:"
    tail = "\nAnswer: ("
    if len(options) <= NARROW:
        lines = [head] + [f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)] + [tail]
        return tok.encode(ctx_text + "".join(lines), add_special_tokens=False)
    ids = list(ctx_ids) + tok.encode(head, add_special_tokens=False)
    for j, o in enumerate(options):
        ids += list(labels.open_ids) + [labels.ids[j]] + tok.encode(f") {o}",
                                                                      add_special_tokens=False)
    return ids + tok.encode(tail, add_special_tokens=False)


def common_prefix(rows: Sequence[Sequence[int]]) -> int:
    """Length of the longest common prefix, at most len(shortest) - 1 (every row keeps a
    suffix to evaluate)."""
    if not rows:
        return 0
    short = min(len(r) for r in rows) - 1
    first = rows[0]
    n = 0
    while n < short and all(r[n] == first[n] for r in rows):
        n += 1
    return n


def readout(label_logits: np.ndarray, temperature: float, top_mask: np.ndarray | None = None
            ) -> list[float]:
    """softmax(logits / T) over the option labels; labels outside the top-n (``top_mask``
    False) get 0 as in the llama.cpp reference; all excluded -> uniform."""
    z = label_logits.astype(np.float64) / float(temperature)
    keep = np.ones(len(z), dtype=bool) if top_mask is None else top_mask
    if not keep.any():
        return [1.0 / len(z)] * len(z)
    e = np.where(keep, np.exp(z - z[keep].max()), 0.0)
    return (e / e.sum()).tolist()


# ---------------------------------------------------------------------- engine
@dataclass
class RowTiming:
    question: str
    tokens: int            # tokens evaluated for this row (after the shared prefix)
    seconds: float         # restore (if any) + eval of the row's suffix + label readout
    restore_seconds: float = 0.0   # of which the rewind / load_state / reset before the row


@dataclass
class Decision:
    """Result of :meth:`Decider.decide`."""
    answers: dict[str, dict[str, Any]]         # decider's shape (rounded), per question id
    probabilities: dict[str, list[float]]      # unrounded, in option order
    rows: dict[str, list[int]] = field(repr=False, default_factory=dict)   # token ids per question
    prefix_tokens: int = 0
    prefix_seconds: float = 0.0
    snapshot_seconds: float = 0.0
    row_timing: list[RowTiming] = field(default_factory=list)
    total_seconds: float = 0.0
    snapshot: str = ""     # "checkpoint" (in-engine), "state" (save_state blob) or "" (none)

    def to_json(self, timing: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"answers": self.answers, "probabilities": self.probabilities}
        if timing:
            out["timing"] = {"prefix_tokens": self.prefix_tokens,
                             "prefix_seconds": round(self.prefix_seconds, 4),
                             "snapshot": self.snapshot,
                             "snapshot_seconds": round(self.snapshot_seconds, 4),
                             "rows": [{"question": r.question, "tokens": r.tokens,
                                       "seconds": round(r.seconds, 4),
                                       "restore_seconds": round(r.restore_seconds, 4)}
                                      for r in self.row_timing],
                             "total_seconds": round(self.total_seconds, 4)}
        return out


def read_decider_config(model_path: str | Path) -> dict[str, Any]:
    """``decider_config.json`` next to the .pq (``<stem>.tokenizer/`` sidecar first)."""
    p = Path(model_path)
    for cand in (p.with_suffix(".tokenizer") / "decider_config.json",
                 p.parent / "decider_config.json"):
        if cand.is_file():
            cfg: dict[str, Any] = json.loads(cand.read_text(encoding="utf-8"))
            return cfg
    return {}


class Decider:
    """Scores typed questions about a state with a decision model loaded in ``model``.

    Args:
        model: A loaded :class:`pinfer.Model` with a tokenizer.
        temperature: Readout temperature; default: ``decider_config.json`` next to the model
            file, else 1.0.
        neutralize_none: Show "none of the above"-style options as "not listed here"
            (default: ``decider_config.json``, else True, as in the reference).
        max_state_tokens: Longest state kept (tokens), as in the reference (16384).
        tokenization: ``"text"`` (default): the row is decoded to text and tokenized again,
            as llama-server receives it; ``"ids"``: decider's ids as built (vLLM engine). They
            differ only for questions with more than 10 options.
        reuse_state: Prefill the rows' common prefix once and restore it for every question.
        top_n: Emulate the llama.cpp reference's top-``n`` probability list (labels outside
            the top ``n`` of the full vocabulary count as 0). Needs the full logits.
        checkpoint: How ``reuse_state`` restores the prefix: None (default) = the engine's
            in-context checkpoint when the library has it (``Model.checkpoint`` / ``rewind``),
            else a state blob; False = always the state blob (``save_state`` /
            ``load_state``); True = the checkpoint (NotImplementedError without it). Both
            paths give the same probabilities bit for bit; the checkpoint is faster and
            copies only the recurrent state, not the prefix's KV rows.
    """

    def __init__(self, model: Model, temperature: float | None = None,
                 neutralize_none: bool | None = None,
                 max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS, tokenization: str = "text",
                 reuse_state: bool = True, top_n: int | None = None,
                 checkpoint: bool | None = None) -> None:
        if model.tokenizer is None:
            raise ValueError("Decider needs a model with a tokenizer")
        if tokenization not in ("text", "ids"):
            raise ValueError("tokenization must be 'text' or 'ids'")
        cfg = read_decider_config(model.path)
        self.model = model
        self.tok = model.tokenizer
        self.temperature = float(temperature if temperature is not None
                                 else cfg.get("temperature", 1.0))
        if not self.temperature > 0:
            raise ValueError("temperature must be > 0")
        self.neutralize_none = bool(cfg.get("neutralize_none", True) if neutralize_none is None
                                    else neutralize_none)
        self.max_state_tokens = int(max_state_tokens)
        self.tokenization = tokenization
        has_ck = bool(getattr(model, "supports_checkpoint", False))   # pinfer >= 0.2 + C API
        if checkpoint and not has_ck:
            raise NotImplementedError("checkpoint=True needs a pinfer with Model.checkpoint "
                                      "and a libpinfer with pinfer_checkpoint")
        self.use_checkpoint = bool(reuse_state) and has_ck and checkpoint is not False
        self.reuse_state = self.use_checkpoint or (bool(reuse_state) and model.supports_state)
        self.top_n = top_n
        self.labels = label_table(self.tok)

    def row_ids(self, state_text: str, q: Question) -> list[int]:
        opts = neutralize_options(q.options) if self.neutralize_none else list(q.options)
        ids = build_row(self.tok, self.labels, state_text, q.text, opts, self.max_state_tokens)
        if self.tokenization == "text":
            ids = self.tok.encode(self.tok.decode(ids), add_special_tokens=False)
        return ids

    def _score(self, suffix: list[int], nopts: int) -> list[float]:
        lab = list(self.labels.ids[:nopts])
        if self.top_n is None:
            return readout(self.model.eval_select(suffix, lab), self.temperature)
        full = self.model.eval(suffix)
        kth = np.partition(full, -self.top_n)[-self.top_n] if self.top_n < len(full) else -np.inf
        sel = full[lab]
        return readout(sel, self.temperature, sel >= kth)

    def decide(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> Decision:
        """Probabilities for every option of every question (one row per question)."""
        t_start = time.perf_counter()
        state_text = render_state(state)
        rqs = {k: render_question(v) for k, v in questions.items()}
        if not rqs:
            raise ValueError("no questions")
        rows = {k: self.row_ids(state_text, q) for k, q in rqs.items()}
        longest = max(len(r) for r in rows.values())
        if longest > self.model.max_ctx:
            raise errors.ContextOverflowError(
                f"a row has {longest} tokens > max_ctx {self.model.max_ctx}", errors.E_CTX_FULL)
        keys = list(rqs)
        lcp = common_prefix([rows[k] for k in keys]) if self.reuse_state and len(keys) > 1 else 0
        res = Decision({}, {}, rows)
        m = self.model
        m.reset()                     # also drops the previous decision's checkpoint
        snap: bytes | None = None
        if lcp > 0:
            t0 = time.perf_counter()
            m.eval_select(rows[keys[0]][:lcp], [])
            res.prefix_tokens, res.prefix_seconds = lcp, time.perf_counter() - t0
            t0 = time.perf_counter()
            if self.use_checkpoint:
                m.checkpoint()
                res.snapshot = "checkpoint"
            else:
                snap = m.save_state()
                res.snapshot = "state"
            res.snapshot_seconds = time.perf_counter() - t0
        for i, k in enumerate(keys):
            t0 = time.perf_counter()
            if i > 0:
                if res.snapshot == "checkpoint":
                    m.rewind()
                elif snap is not None:
                    m.load_state(snap)
                else:
                    m.reset()
            t_restore = time.perf_counter() - t0
            suffix = rows[k][lcp:]
            p = self._score(suffix, len(rqs[k].options))
            res.row_timing.append(RowTiming(k, len(suffix), time.perf_counter() - t0, t_restore))
            res.probabilities[k] = p
            res.answers[k] = format_answer(rqs[k], p)
        res.total_seconds = time.perf_counter() - t_start
        return res
