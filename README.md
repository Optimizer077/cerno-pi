# cerno-pi — Cerno decision models on a Raspberry Pi 4

**Status: work in progress.** The first verified version (correctness against the reference engine,
latency on the Pi) is being finished; this README will carry the numbers when it lands.

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

## Weights

The model weights are **not** included and are never redistributed here. Download them from the
Hugging Face model cards above under their own terms. The model cards declare Apache-2.0, but the
weight repositories carry no LICENSE file of their own.

## License

The code in this repository is Apache-2.0 ([LICENSE](LICENSE)); the adapted parts and their origins
are listed in [NOTICE](NOTICE).
