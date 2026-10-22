# Changelog

Newest first. Measurements and their result files: [CERNO.md](CERNO.md).

## 2026-10-22 (KST)

### Paused at a checkpoint (14:55)
- Work paused for an overclock test on the shared Pi; status, open items and next steps in
  CERNO.md §8. All timings so far are at 1.5 GHz.

### Added
- `cerno_pi` package (depends on pinfer ≥ 0.2.0 with `Model.eval_select`): `Decider` reproduces
  the scoring protocol of Cerno's llama.cpp engine (decider's state-first rows, label readout,
  calibration temperature from `decider_config.json`, "none of the above" neutralisation),
  prefills the rows' shared state once and restores a snapshot per question, and reads only the
  label rows of the output head. CLI: `cerno-pi decide` / `python -m cerno_pi decide`.
- `bench/ref_llama.py` (runs Cerno's `engine_llama.py` unmodified against llama-server and records
  unrounded probabilities and token ids), `bench/decide_bench.py` (Pi timing through piquant's
  harness), `bench/compare.py`, `bench/compare_refs.py`, `bench/latency_table.py`,
  `scripts/build_llama_server.sh`, `requests/correctness.json` (18 requests, 52 questions).
- `tests/test_decide.py`: question rendering, neutralisation, prefix logic, the readout formula
  equals the reference's `p^(1/T)` renormalisation, answer shapes, prompt layout with the Cerno
  tokenizer (6/6).

### Measured (Pi 4 at 1.5 GHz)
- Prefill 5.53 / 5.57 tok/s (64 / 256 tokens); a decision on a short state 17–38 s (median 32.6 s),
  later questions of the same state ~6.7 s each; peak RSS 2.32 GiB.
- Correctness on 31 rows: token ids identical to llama-server's on all rows; max |Δp| 0.0108,
  mean 0.0028, argmax 30/31 (one 0.497 / 0.503 tie) — the same spread llama.cpp shows between
  its own f16-KV and f32-KV settings (max 0.0092, mean 0.0027).
