# cerno-pi — Cerno decision models on a Raspberry Pi 4

**Status: work in progress (paused 2026-10-22).** Cerno 4B runs on the Pi 4 and matches the
reference engine within llama.cpp's own numeric spread on the rows measured so far; the comparison
with llama.cpp's speed on the Pi is still to come. Details and open items: [CERNO.md](CERNO.md).

cerno-pi runs **Cerno 4B**, a decision model by Apertix, on a Raspberry Pi 4 (4 GB) with
[pinfer](https://github.com/Optimizer077/piquant), a Pi-4-native inference engine for the Qwen3.5
model family. A Cerno model reads a state and typed questions (multiple choice, yes/no, scored scales)
and returns calibrated probabilities per option in one forward pass, without generating text.

## Credit: the original work

Cerno is **not** our work. All credit for the models, their training, the decision protocol and the
evaluation belongs to their authors:

- **Cerno models and inference code** — Apertix:
  [github.com/apertixsk/cerno-inference](https://github.com/apertixsk/cerno-inference) (Apache-2.0),
  model cards [apertixai/cerno-4b](https://huggingface.co/apertixai/cerno-4b) and
  [apertixai/cerno-4b-GGUF](https://huggingface.co/apertixai/cerno-4b-GGUF) on Hugging Face.
  Cerno 4B is built on Qwen3.5-4B (Qwen team, Alibaba).
- **Prompt layout and typed questions** — [decider](https://github.com/Mapika/decider) (Apache-2.0),
  as vendored in cerno-inference.

This repository is an independent port of their scoring protocol to the Pi, with the exact commits it
was derived from listed in [NOTICE](NOTICE). It is not affiliated with or endorsed by Apertix, the
decider authors, TypeSafe AI or the Decision Index maintainers.

## What has been verified so far

Raspberry Pi 4 (4 GB), Cortex-A72 at **1.5 GHz**, 4 threads, the published Q4_0 GGUF converted
losslessly (every 4-bit and 8-bit block copied bit for bit):

| | result |
|---|---|
| fits in memory | 2.32 GiB peak RSS during decisions (the Pi has 3.7 GiB) |
| prefill | 5.5 tokens/s (64 and 256 tokens) |
| one decision (2–5 questions on a short state) | 17–38 s, median 32.6 s; first question ~17 s, each further question ~7 s |
| agreement with Cerno's own llama.cpp engine on the same GGUF (31 question rows) | identical tokens on every row; max \|Δp\| 0.011, mean 0.0028; same top answer on 30/31 (the other is a 0.497 / 0.503 tie) |
| llama.cpp against itself (f16 vs f32 KV cache, 52 rows) | max \|Δp\| 0.0092, mean 0.0027 |

All result files are in [results/](results/); how each number was taken is in [CERNO.md](CERNO.md).

## How it works

1. `tools/gguf_to_pq.py` of pinfer converts the GGUF into pinfer's `.pq` format without
   re-quantizing (llama.cpp's Q4_0 / Q8_0 blocks are pinfer's Q4_G32 / Q8_G32 groups), and checks
   the result bit for bit (`--verify`).
2. `cerno_pi.Decider` renders one prompt row per question exactly as Cerno's llama.cpp engine does
   (decider's state-first layout), prefills the state that all rows share **once**, snapshots it,
   and scores every question from the snapshot.
3. Only the option-label rows of the 248k-row output head are computed (`Model.eval_select`), and
   the probabilities are `softmax(label logits / T)` with the model's calibration temperature
   (Cerno 4B: 1.534, from `decider_config.json`).

## Usage

```bash
# 1. pinfer (github.com/Optimizer077/piquant) with Model.eval_select, built for the Pi
# 2. weights: download cerno-4b-v1.1-q4_0.gguf (apertixai/cerno-4b-GGUF) and the tokenizer files +
#    decider_config.json (apertixai/cerno-4b), then convert on a PC:
python tools/gguf_to_pq.py cerno-4b-v1.1-q4_0.gguf cerno-4b-q4_0.pq --tokenizer cerno-4b/ --verify
# 3. decide (one JSON line per request):
cerno-pi decide -m cerno-4b-q4_0.pq --request requests/correctness.json --timing
cerno-pi decide -m cerno-4b-q4_0.pq --state "rm -rf /tmp/build-cache" \
    --questions '{"destructive": {"type": "noul", "instructions": "Would this delete data?"}}'
```

```python
import pinfer
from cerno_pi import Decider

with pinfer.Model.load("cerno-4b-q4_0.pq") as m:          # finds cerno-4b-q4_0.tokenizer/
    d = Decider(m)
    r = d.decide("hey do u have showers at the downtown location", {
        "kind": {"type": "choice", "instructions": "What kind of message is this?",
                 "criteria": {"billing": "invoices, charges, refunds", "other": None}},
        "needs_human": {"type": "noul", "instructions": "Does this need a human?"}})
    print(r.answers)            # decider's answer format; r.probabilities: unrounded
```

Question types: `choice` (criteria = option names, optionally with descriptions; up to 255),
`noul` (yes/no), `score` (2–10 ordered levels). Options: `--no-reuse` (no snapshot),
`--tokenization ids` (decider's ids instead of re-tokenized text; differs only above 10 options),
`--top-n 260` (emulate the reference's top-260 probability list; reads the full head).

## Layout

```
cerno_pi/decide.py        prompt rows, readout, shared-state snapshot (ported protocol, see NOTICE)
cerno_pi/cli.py           cerno-pi decide
bench/ref_llama.py        reference: Cerno's engine_llama.py, unmodified, against llama-server
bench/decide_bench.py     timing on the Pi (through piquant's harness)
bench/compare.py          pinfer vs reference, row by row
requests/correctness.json the test requests (written for this project, no benchmark data)
results/                  the result files cited in CERNO.md
CERNO.md                  lab notebook: sources, revisions, licence, every measurement
```

## Weights

The model weights are **not** included and are never redistributed here. Download them from the
Hugging Face model cards above under their own terms. The model cards declare Apache-2.0, but the
weight repositories carry no LICENSE file of their own.

## License

The code in this repository is Apache-2.0 ([LICENSE](LICENSE)); the adapted parts and their origins
are listed in [NOTICE](NOTICE).
