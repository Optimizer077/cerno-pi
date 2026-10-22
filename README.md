# cerno-pi — Cerno decision models on a Raspberry Pi 4

cerno-pi runs **Cerno 4B**, a decision model by Apertix, on a Raspberry Pi 4 (4 GB) with
[pinfer](https://github.com/Optimizer077/nibblepi), the Pi-4-native inference engine of the nibblepi
project for the Qwen3.5 model family. A Cerno model reads a state and typed questions (multiple
choice, yes/no, scored scales) and returns calibrated probabilities per option in one forward pass,
without generating text.

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

## Results

Raspberry Pi 4 (4 GB), Cortex-A72 at **1.5 GHz**, 4 threads, the published Q4_0 GGUF. pinfer reads it
converted losslessly (every 4-bit and 8-bit block copied bit for bit); llama.cpp reads the GGUF itself.
Both engines run the same prompts.

| on the Pi 4 | pinfer + cerno-pi | llama.cpp `llama-server` (Cerno's engine protocol) |
|---|---|---|
| prefill speed | **6.4–6.6 tokens/s** | 2.43–2.45 tokens/s (`llama-bench`, 64 / 256 tokens) |
| one decision, 8 requests both engines ran (26 questions) | **median 27.6 s** | median 112.4 s (4.5× longer in total) |
| first question of a request | median 12.0 s | median 33.7 s |
| each further question on the same state | **median 5.4 s** (the state is reused) | median 33.6 s (the whole row again) |
| peak memory (VmHWM) | **2395 MiB** | 3554 MiB |
| agreement with Cerno's llama.cpp engine on a PC (52 questions) | same tokens on every row; max \|Δp\| 0.011, same top answer on 51/52 | the Pi's llama.cpp: max \|Δp\| 0.015, same top answer on 24/26 |

On all 18 requests of the test set, pinfer answers a request in 13.6–89.3 s (median 27.7 s): the
first question takes a median 14.4 s, each further question 5.5 s. llama.cpp re-reads the whole
prompt for every question because its prompt cache cannot reuse the state of this hybrid
(recurrent + attention) model at the point where the rows diverge. pinfer saves the state once and
restores it. Per-request tables, every result file and how each number was taken: [CERNO.md](CERNO.md).

## What a decision looks like

One request of the test set, as Jev / decider-style input: a state and typed questions.

```python
state = "hey do u have showers at the downtown location"
questions = {
    "kind": {"type": "choice", "instructions": "What kind of message is this?",
             "criteria": {"incident": "an outage or something broken in production",
                          "billing": "invoices, charges, refunds", "account": "cancellations, plan changes, access",
                          "feedback": "praise or suggestions", "spam": "unsolicited or scam", "other": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this message?",
                "criteria": ["no action needed", "can wait a few days", "should be handled today",
                             "needs attention right now"]},
    "tone": {"type": "score", "instructions": "What is the sender's tone?",
             "criteria": ["hostile", "negative", "neutral", "positive"]},
    "needs_human": {"type": "noul",
                    "instructions": "Does this message need a human to respond rather than an automated reply?"},
}
```

The output is a probability for every option of every question. pinfer on the Pi is shown next to
Cerno's own llama.cpp engine on the same GGUF:

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="docs/figures/decision_example-mobile-dark.png">
  <source media="(max-width: 600px)" srcset="docs/figures/decision_example-mobile.png">
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/decision_example-dark.png">
  <img alt="Grouped bars for the four questions of the support-triage request. kind: other 0.612 (pinfer) vs 0.618 (reference), account 0.159 vs 0.154, feedback 0.102 vs 0.102, billing 0.062 vs 0.061, incident and spam about 0.03. urgency: no action needed 0.462 vs 0.462, can wait 0.319 vs 0.320, today 0.182 vs 0.182, right now 0.037 vs 0.036. tone: neutral 0.840 vs 0.843, positive 0.111 vs 0.108. needs_human: no 0.497 vs 0.495, yes 0.503 vs 0.505." src="docs/figures/decision_example.png">
</picture>

| question | option | pinfer on the Pi | reference (llama.cpp) |
|---|---|---|---|
| kind | incident | 0.0360 | 0.0357 |
|  | billing | 0.0615 | 0.0615 |
|  | account | 0.1591 | 0.1542 |
|  | feedback | 0.1023 | 0.1015 |
|  | spam | 0.0286 | 0.0288 |
|  | **other** | **0.6125** | **0.6183** |
| urgency | **0: no action needed** | **0.4622** | **0.4615** |
|  | 1: can wait a few days | 0.3188 | 0.3199 |
|  | 2: should be handled today | 0.1825 | 0.1823 |
|  | 3: needs attention right now | 0.0365 | 0.0363 |
| tone | 0: hostile | 0.0153 | 0.0152 |
|  | 1: negative | 0.0341 | 0.0338 |
|  | **2: neutral** | **0.8401** | **0.8428** |
|  | 3: positive | 0.1106 | 0.1082 |
| needs_human | no | 0.4968 | 0.4950 |
|  | **yes** | **0.5032** | **0.5050** |

## How it works

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="docs/figures/decision_flow-mobile-dark.gif">
  <source media="(max-width: 600px)" srcset="docs/figures/decision_flow-mobile.gif">
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/decision_flow-dark.gif">
  <img alt="Animation of one request on the Pi (a contract clause, four questions), with the Pi's real times played four times faster. The 118-token state is prefilled once (17.3 s) and saved as a snapshot (0.26 s). The first question runs on the prefilled state, each later one restores the snapshot; each runs only its own 23–48 tokens and reads only the option-label rows of the 248,320-row head (3.7–6.9 s), and its probabilities fill in as bars: 90 days 0.97, data returned yes 0.88, confidentiality survives yes 0.94, favours balanced 0.56. Total 37.2 s." src="docs/figures/decision_flow.gif">
</picture>

1. nibblepi's `tools/gguf_to_pq.py` converts the GGUF into pinfer's `.pq` format without
   re-quantizing (llama.cpp's Q4_0 and Q8_0 blocks are pinfer's Q4_G32 and Q8_G32 groups) and
   checks the result bit for bit (`--verify`). The model was trained for the Q4_0 grid, so that grid
   must stay exactly as published.
2. `cerno_pi.Decider` builds one prompt row per question, exactly as Cerno's llama.cpp engine does
   (decider's state-first layout). It prefills the state that all rows share **once**, snapshots
   the model state, and scores every question from the snapshot. The results are bit-identical to
   scoring each row from scratch (checked on the Pi).
3. Only the option-label rows of the 248,320-row output head are computed (`Model.eval_select`;
   −649 MiB of memory). The probabilities are `softmax(label logits / T)` with the model's
   calibration temperature (Cerno 4B: 1.534, from `decider_config.json`).

## Agreement over the whole test set

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="docs/figures/agreement-mobile-dark.png">
  <source media="(max-width: 600px)" srcset="docs/figures/agreement-mobile.png">
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/agreement-dark.png">
  <img alt="Left: pinfer's probability against the reference's for all 213 option probabilities of the 52 questions; every point lies on the diagonal. Right: the differences magnified, within ±0.011 for pinfer, the same size as llama.cpp's own differences between its f32 and f16 KV cache settings (within ±0.0092)." src="docs/figures/agreement.png">
</picture>

The differences come from numerics, not from the protocol: both engines see the same token ids on
every row (52/52) and the same weights. The remaining differences come from rounding in the int8
activations and the attention / recurrent kernels. llama.cpp differs from itself by as much when
only its KV-cache precision changes.

## Dataset

- **What was evaluated:** [requests/correctness.json](requests/correctness.json): **18 requests with
  52 typed questions** (19 multiple choice, 21 yes/no, 12 scored scales; 213 options; two choices
  with 16 and 24 options use the wide label set). The categories are:
  - support triage: the two requests of Cerno's README examples, billing;
  - agent guardrails for tool calls, as a shell command and as a JSON tool call;
  - review sentiment, smart-home intents, request routing;
  - country and language identification (German text);
  - a delivery note in French with an emoji and Japanese;
  - a JSON state with a 9-ticket array;
  - a contract clause, natural-language inference, arithmetic;
  - phishing;
  - a 400-token incident log;
  - medical triage, code review, translation quality.
- **How it was written:** by hand for this port, with Claude Code (AI-assisted). It contains no
  benchmark data. It is a correctness and latency set, not a quality benchmark: the reference
  answers are not labels but the outputs of Cerno's own engine on the same GGUF.
- **License:** ours, Apache-2.0, like the rest of this repository.
- **Why not Cerno's `sample/` set or the Decision Index:** cerno-inference ships only its answers on
  the 60 sampled requests, not the requests. Drawing the same requests needs the staged Decision
  Index 0.3 suite: about 7 GB of downloads from 43 datasets under their own terms, one of them gated
  on Hugging Face.
- **Files:** the requests are in `requests/`. The reference outputs (`results/ref_llama_pc*.jsonl`)
  and every Pi result file are in `results/`.

For the official evaluation of the models, see the
[Decision Index](https://github.com/apolinario/decision-index) and the model cards
([cerno-4b](https://huggingface.co/apertixai/cerno-4b): public index 57.60 with FP8 weights;
[cerno-4b-GGUF](https://huggingface.co/apertixai/cerno-4b-GGUF): 57.23 through llama.cpp).

## Usage

```bash
# 1. pinfer 0.2.0 from github.com/Optimizer077/nibblepi (with Model.eval_select), on the Pi.
# 2. Weights (not included here): cerno-4b-v1.1-q4_0.gguf from apertixai/cerno-4b-GGUF, and the
#    tokenizer files + decider_config.json from apertixai/cerno-4b. Convert on a PC (numpy only):
python tools/gguf_to_pq.py cerno-4b-v1.1-q4_0.gguf cerno-4b-q4_0.pq --tokenizer cerno-4b/ --verify
# 3. This package (pinfer is not on PyPI: install its wheel first), then decide:
pip install .
cerno-pi decide -m cerno-4b-q4_0.pq --request requests/correctness.json --timing
cerno-pi decide -m cerno-4b-q4_0.pq --state "rm -rf /tmp/build-cache" \
    --questions '{"destructive": {"type": "noul", "instructions": "Would this delete data?"}}'
```

```python
import pinfer
from cerno_pi import Decider

with pinfer.Model.load("cerno-4b-q4_0.pq") as m:          # finds cerno-4b-q4_0.tokenizer/
    d = Decider(m)                                        # temperature from decider_config.json
    r = d.decide("hey do u have showers at the downtown location", {
        "kind": {"type": "choice", "instructions": "What kind of message is this?",
                 "criteria": {"billing": "invoices, charges, refunds", "other": None}},
        "needs_human": {"type": "noul", "instructions": "Does this need a human?"}})
    print(r.answers)            # decider's answer format; r.probabilities: unrounded
```

Question types:
- `choice`: criteria = option names, optionally with descriptions; up to 255 options;
- `noul`: yes/no;
- `score`: 2–10 ordered levels.

Options:
- `--no-reuse`: no snapshot;
- `--tokenization ids`: decider's ids instead of re-tokenized text; differs only above 10 options;
- `--top-n 260`: emulate the reference's top-260 probability list; reads the full head.

## Layout

```
cerno_pi/decide.py        prompt rows, readout, shared-state snapshot (ported protocol, see NOTICE)
cerno_pi/cli.py           cerno-pi decide
bench/ref_llama.py        reference: Cerno's engine_llama.py, unmodified, against llama-server
bench/decide_bench.py     pinfer timing on the Pi (through nibblepi's harness)
bench/llama_pi.py         llama-server timing on the Pi with the reference prompts
bench/compare*.py         pinfer vs reference, reference vs itself, run vs run
requests/correctness.json the test requests (written for this project, no benchmark data)
results/                  the result files cited in CERNO.md
tools/                    README figures (figstyle.py from nibblepi)
CERNO.md                  lab notebook: sources, revisions, licence, every measurement
```

## Weights

The model weights are **not** included and are never redistributed here. Download them from the
Hugging Face model cards above under their own terms. The model cards declare Apache-2.0, but the
weight repositories carry no LICENSE file of their own.

## License

The code in this repository is Apache-2.0 ([LICENSE](LICENSE)); the adapted parts and their origins
are listed in [NOTICE](NOTICE).
