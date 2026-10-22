# Cerno 4B on the Raspberry Pi 4 — lab notebook

Owner: Cerno agent, 2026-10-22 (KST). Engine: pinfer (`github.com/Optimizer077/piquant`), branch
`worktree-agent-a5c9f7577a224b708` from main `641a683`. Question asked by the owner: can Cerno 4B
be set up, tested carefully and optimized on the Pi 4?

**Status at the checkpoint (14:55 KST, paused for the lead's overclock test).**
Yes, it runs, and it is correct within llama.cpp's own numeric spread:

- memory: 2.32 GiB peak RSS for a decision run (of 3794 MiB), 2.6 GiB when the full head is read;
- latency at 1.5 GHz: 5.5 tokens/s of prefill, so a 2–5-question decision on a short state takes
  17–38 s (median 32.6 s), the first question ~17 s and each further question ~7 s;
- accuracy: on 31 question rows, the same token ids as llama-server on every row, max |Δp| 0.011
  and mean |Δp| 0.0028 against Cerno's own llama.cpp engine on the same GGUF, argmax agreement
  30/31 (the one disagreement is a 0.497 / 0.503 tie). llama.cpp differs from itself by the same
  amount (max 0.0092, mean 0.0027) between two of its own KV / attention settings.

What is still open is listed in §8.

## 1. Sources, revisions, licence

| Item | Value |
|---|---|
| Inference code | `github.com/apertixsk/cerno-inference` @ `9941548d836ff53b56e326860f94b93adc79bee9` (2 commits, Apache-2.0), cloned to `E:\Personal_tries\third_party\cerno-inference` (outside every repo); vendors decider `Mapika/decider` @ `1c15a48f199adc0eb1952b17441516b5358d79a0` (Apache-2.0) |
| Decision Index kit | `github.com/apolinario/decision-index` @ `9eb2dbe2a358004c8782c66e40a83ac07b953fec` (MIT), cloned to `E:\Personal_tries\third_party\decision-index`; only `decision_index.engines.base` is used (engine base class) |
| GGUF | `apertixai/cerno-4b-GGUF` @ `61c1bc809b113848001627e6d6ae8bfa1d64daee`, `cerno-4b-v1.1-q4_0.gguf`, 3,373,287,808 B, sha256 `93a94cc5216efd09f6116fbc3f50ec2a621cb7c3bb4b1a8da71734fd6ad25eca` (= the repo's `SHA256SUMS`), md5 `038964105467d52a388591e819d16a67` |
| Tokenizer + config | `apertixai/cerno-4b` @ `ed4b5c59b542ee96c227ee874776c973980c640b`: `tokenizer.json` sha256 `06b95093…a0e523`, `tokenizer_config.json` `171ecbe7…`, `config.json` `35b5c20e…`, `decider_config.json` `bfff899a…` (temperature **1.534**), `chat_template.jinja` `a4aee8af…`. The FP8 weights (`model.safetensors`, 4.84 GB, sha256 `392544403c…`) were **not** downloaded; a few small tensors (A_log, dt_bias, norms of layer 0) were read with HTTP range requests for cross-checks |
| Local copies | piquant main checkout, gitignored: `models/cerno-4b/gguf/`, `models/cerno-4b/hf/`, `models/cerno-4b/cerno-4b-q4_0.pq` (md5 `7d0c7ca610490eddfedba6095726744a`) + `cerno-4b-q4_0.tokenizer/` |
| On the Pi | `~/pinfer/models/cerno-4b-q4_0.pq` + `.tokenizer/` (md5 checked on the Pi), `~/pinfer/engine/pinfer_run_cerno1` (md5 `34f2946a72a81ecf4f5522863e8ceeee`), `~/pinfer/tmp/cerno/` (libpinfer.so md5 `81855186830be1d15a616285196b4bb9`, Python sources, requests); listed with undo in piquant `docs/pi-state.md` |

**Weights licence.** The model cards of `apertixai/cerno-4b` and `apertixai/cerno-4b-GGUF` declare
`license: apache-2.0` (YAML and a "Licence and credits" section: "Apache-2.0, like the FP8 weights
it was made from (base model Qwen3.5-4B-Base, Apache-2.0)"), but neither weight repository has a
LICENSE file of its own (the FP8 repo's `inference/LICENSE` is the code's). This project uses the
weights only locally for testing; no weight, GGUF or `.pq` is committed or redistributed.

Third-party code was read in full before anything ran; it was run only from a separate uv venv
(Python 3.12, transformers 4.57.6, torch 2.14.1+cpu, tokenizers 0.22.2, decision-index 0.3 from
the clone) with `python -I`, from outside the clone directories.

## 2. The model

From the GGUF metadata (and `config.json`): Qwen3.5 hybrid, hidden 2560, 32 layers (full attention
at 3, 7, …, 31; mask `0x88888888`), ffn 9216, 16 query / 4 KV heads × 256 (gated, RoPE on 64
dims, θ 1e7), linear attention 16 K heads / 32 V heads × 128, conv 4, vocab 248,320, eps 1e-6,
context 262,144, **no MTP** (GGUF has no `nextn` block). Tensors: body Q4_0 2007.6 MB,
`token_embd` Q8_0 675.4 MB, `output` Q8_0 675.4 MB, F32 3.8 MB.

Findings that the converter has to respect (piquant `engine/FORMAT.md`, "Importing a llama.cpp GGUF"):
- **Untied head.** `config.json` says `tie_word_embeddings: true`, but in the GGUF `output.weight`
  differs from `token_embd.weight` in 111,621 of 248,320 rows (correlation 0.99999): the QAT
  build trained them apart. The engine needed `lm_head.weight` support.
- **V heads in llama.cpp's tiled order** (`_LinearAttentionVReorderBase`, llama.cpp
  `conversion/qwen.py`): GGUF `dt_bias` equals the HF checkpoint's exactly only after tiling;
  pinfer uses HF's grouped order (`kh = h / rep`), so the converter un-tiles.
- `ssm_a` = −exp(A_log) computed by llama.cpp's converter; equal to −exp(HF A_log) within 1 ulp.
- The GGUF is **not** the FP8 checkpoint re-quantized: its norm weights differ from `1 + w` of
  the FP8 repo by up to 4e-3 (not bf16-representable), its conv weights by 1e-3, one dt_bias
  value (layer 25) by 1.1e-4. So the GGUF itself is the model to reproduce, and the reference
  is llama.cpp on that GGUF.
- Q8_0 codes are in [−127, 127] (no −128), all scales positive and finite.

**Memory budget** (Pi 4: 3794 MiB, 3548 MiB available at idle): weights touched per forward
1915 MiB (body) + the label tiles of the head + a few embedding rows; recurrent state 24 × 32 ×
128 × 128 × 4 B = 48 MiB, conv state 2.3 MiB, KV 64 KiB per token (8 layers × 4 heads × 256 × 2
× fp32), so a 500-token row adds 31 MiB; a state snapshot is ~51 MiB + KV. Measured: 2627.6 MiB
peak RSS with the full head (`pinfer_run`), **2323.7 MiB** for a decision run with `eval_select`
(Python + tokenizer included). Fits with ~1.2 GiB headroom.

## 3. The decision protocol (what `cerno_pi.decide` reproduces)

Read from `engine_llama.py`, `decider/{prompt,systemone,infer}.py` (the vLLM engine uses the same
rows and readout):
- one row per question (`plan_rows(rqs, isolated=False)`): `"Context:\n" + state` +
  `"\n\nQuestion: <instructions>\nOptions:\n(A) <opt>\n…\nAnswer: ("`; ≤ 10 options are
  lettered A..J and the whole row is tokenized as one string (Cerno's `JOINT_FIRST_PIECE`);
  more options get one label token each (A..Z then two-letter single tokens, 255 in all) and are
  built piecewise; states are cut at 16,384 tokens (token level, then decoded);
- JSON states: `json.dumps(annotate_indices(state), ensure_ascii=False)` (arrays of ≥ 8 elements
  get `_index`); choice options `"name: description"`; score levels `"i: description"`; yes/no
  `no` / `yes` (+ descriptions); "none of the above"-like options are shown as "not listed here";
- the llama.cpp engine decodes the row to text and llama-server tokenizes it again (no BOS: the
  `/tokenize` ids equal the HF tokenizer's for all 52 rows here); the readout is the server's
  top-260 probabilities of the full-vocabulary softmax, the label tokens' entries renormalised
  after `p^(1/T)`. That is `softmax(logits_labels / T)`, except that a label outside the top 260
  counts as 0 (never the case on this set: every label of every row was in the top 260).

`cerno_pi.decide.Decider` builds the same rows (default `tokenization="text"`, as llama-server
sees them), computes only the label logits (`Model.eval_select`), and applies
`softmax(l / T)`; `top_n=260` emulates the truncation (needs the full head).

## 4. Converter exactness (piquant `tools/gguf_to_pq.py --verify`)

| Check | Result |
|---|---|
| tensors | 427 (250 quantized, 177 F32), 192 of them permuted (24 linear layers × 8) |
| quantized weights | 4,840,488,960: every fp16 scale and every code equal to the GGUF's after the permutation; dequantized weights equal bit for bit |
| F32 values | 961,536 equal bit for bit |
| reference packers | `tools/quant.py` `pack_q4` / `pack_q8` rebuild all 3,358,474,240 quantized bytes |
| permutation direction | checked independently against the HF checkpoint: un-tiled `dt_bias` equal in 23 of 24 layers (layer 25: one value 1.1e-4 off, see §2), A = −exp(A_log) within 1 ulp |
| run | 2:17 min, 7.1 GB peak RSS (WSL); re-run `--verify-only` with the committed code: same result |

## 5. Correctness against the reference (Cerno's llama.cpp engine, same GGUF)

Reference: `bench/ref_llama.py` runs `LlamaGGUFEngine` unmodified against llama-server v0.7.0
@ `620d222` (x86 host build, `-c 16384 -t 8 --parallel 1`, defaults otherwise: f16 KV, flash
attention auto), on `requests/correctness.json` (18 requests / 52 questions written for this
test: the two README examples plus triage, guardrails, sentiment, intents, 16- and 24-option
choices, a JSON state with a 9-element array, French / emoji / Japanese text, a contract clause,
NLI, arithmetic, phishing, a 400-token incident log, medical triage, code review, translation
quality; no benchmark data). Results: `results/ref_llama_pc.jsonl`.

pinfer: `bench/decide_bench.py` on the Pi (NEON build, 4 threads, fp32 KV, state reuse,
`eval_select`), set A = 12 requests / 31 rows. `bench/compare.py`:

| comparison | rows | token ids identical | max \|Δp\| | mean \|Δp\| | median | argmax agree | rows > 1e-3 | file |
|---|---|---|---|---|---|---|---|---|
| pinfer (Pi) vs llama.cpp default | 31 | 31 | 0.0108 | 0.0028 | 0.0020 | 30 / 31 | 25 | `results/compare_reuse_a.json` |
| pinfer (Pi) vs llama.cpp f32 KV, no flash attention | 31 | 31 | 0.0113 | 0.0033 | 0.0024 | 30 / 31 | 24 | `results/compare_reuse_a_vs_kvf32.json` |
| llama.cpp default vs llama.cpp f32 KV, no FA (all 52 rows) | 52 | — | 0.0092 | 0.0027 | 0.0021 | 52 / 52 | 31 | `bench/compare_refs.py` on the two ref files |

The only argmax disagreement: `sentiment_review.price_complaint` (llama 0.497 / 0.503, pinfer
0.508 / 0.492). Example: `support_triage.kind` llama `[0.0357 0.0615 0.1542 0.1015 0.0288 0.6183]`,
pinfer `[0.0360 0.0615 0.1591 0.1023 0.0286 0.6125]`.

**Reading.** The differences exceed the 1e-3 the lead asked to investigate, but switching
llama.cpp's own KV cache from f16 to f32 and flash attention off moves its answers by the same
amount, and does not bring pinfer closer. So they are not a KV-precision artefact, and nothing
points to a semantic difference (same tokens, same layout, same weights bit for bit). Known
numeric differences that remain: llama.cpp rounds each activation block's scale to fp16 and uses
`1/d` instead of `127/amax` (pinfer keeps fp32 scales), different summation order, and the
recurrent / attention kernels. Not yet done (§8): pinfer with fp16 KV, the ARM llama.cpp on the
Pi as a second reference, and the remaining 21 rows (set B) on the Pi.

## 6. Pi measurements (all at **1.5 GHz**: `freq_khz` 1500000 on all four cores before and after every run, throttled 0x0)

| run | what | result | file |
|---|---|---|---|
| 14:28 `run --lock` (cold page cache) | `pinfer_run_cerno1`, 64 tokens | prefill 3.34 tok/s (first touch from the SSD), peak RSS 2627.1 MB | piquant `logs/runs/` (not a bench file) |
| 14:32 `cerno-prefill` | `pinfer_run_cerno1 --bench`, 64 / 256 tokens, 4 threads, `-b` 64 | prefill **5.53 / 5.57 tok/s**, decode 1.41 / 1.39 tok/s, peak RSS 2627.6 / 2644.6 MB | `results/cerno-prefill_20261022_143214.json` |
| 14:39 `cerno-decide-reuse-a` | 12 requests / 31 questions, state reuse, `eval_select`, after one warm-up decision | decision median **32.6 s** (mean 28.7, p80 34.7, p95 36.2, max 38.3); first question median 17.0 s; later questions median **6.7 s** (n 19); 5.51 tok/s over all tokens evaluated; peak RSS **2323.7 MiB** | `results/cerno-decide-reuse-a_20261022_143916.json` |

Per request (set A; `bench/latency_table.py`):

| request | questions | shared prefix tok | row tok (sum) | decision s | first q s | later q s (mean) |
|---|---|---|---|---|---|---|
| support_triage | 4 | 15 | 193 | 38.3 | 15.7 | 7.5 |
| agent_guardrail | 4 | 32 | 158 | 34.7 | 11.3 | 7.8 |
| billing_angry | 3 | 46 | 137 | 33.3 | 20.8 | 6.2 |
| agent_guardrail_email | 3 | 71 | 128 | 36.2 | 18.2 | 9.0 |
| sentiment_review | 4 | 40 | 147 | 33.8 | 12.2 | 7.2 |
| intent_smart_home | 2 | 26 | 70 | 17.4 | 13.1 | 4.3 |
| router_wide | 2 | 37 | 140 | 32.0 | 22.4 | 9.6 |
| country_wide | 2 | 39 | 147 | 33.8 | 27.0 | 6.7 |
| unicode_mix | 2 | 56 | 79 | 24.5 | 20.4 | 4.0 |
| nli | 1 | 0 | 90 | 16.1 | 16.1 | — |
| math_word | 2 | 43 | 71 | 20.7 | 17.1 | 3.6 |
| medical_triage | 2 | 38 | 94 | 23.8 | 16.9 | 6.9 |

The workload is compute-bound prefill at ~5.5 tok/s, flat from 64 to 256 tokens, so latency is
tokens / 5.5 s plus ~50 ms per snapshot restore.

## 7. Optimizations so far

| optimization | effect | evidence |
|---|---|---|
| shared-state snapshot (prefill the rows' common prefix once, `save_state` / `load_state` per question) | tokens evaluated on the whole correctness set 7313 → 3703 (−49 %); on the 400-token incident log (5 questions) 2137 → 589 (3.6× less work). Pi time scales with tokens (5.5 tok/s), so ≈ 22 → 11 min for the set | token counts from `results/ref_llama_pc.jsonl` rows; the no-reuse Pi run is open (§8) |
| label rows only (`pinfer_eval_select`: 4-row tiles of the 255 labels instead of the 248,320-row Q8_0 head) | peak RSS 2627.6 → 2323.7 MiB (−304 MiB; the full head is 675 MB of Q8_0 that is never paged in); per question ~0.17 s less head streaming (675 MB at ~4 GB/s; not separately timed yet) | `cerno-prefill` vs `cerno-decide-reuse-a` |
| converter keeps the QAT grid exactly (no re-quantization) | quality of the published GGUF (Decision Index 57.23 per its card) kept bit for bit | §4 |

## 8. Open items and next steps (in order)

1. **Pi gates for the engine change** (piquant): `python harness/pi.py bench cerno-gate-spec-exact
   -- bash engine/spec_exact.sh ./engine/pinfer_run_cerno1` (20/20 `match`) and
   `engine/bench/ctx_check.sh` (5/5 identical to `pinfer_run_fast`). The binary is on the Pi
   (md5 `34f2946a…`). Host gates passed (golden / spec / state, pytest 126 + 3, malformed 43/43).
2. **Correctness set B** on the Pi (json_tickets, contract_clause, phishing, long_incident,
   code_review, translation_quality; 1806 tokens ≈ 5.5 min at 1.5 GHz): `bench cerno-decide-reuse-b`
   with the `--only` list; then `bench/compare.py results/ref_llama_pc.jsonl <a> <b>` for all 52 rows.
   The staged `tmp/cerno/requests/correctness.json` on the Pi still has the earlier phishing text
   (domain `.com`; changed to `.example` here before committing) — push the new file first.
3. **No-reuse timing** (`--no-reuse`, e.g. long_incident + support_triage, ≈ 7 min) to measure
   the snapshot gain directly; `--top-n 260` run on a subset to time the full head (and to
   confirm the top-n emulation changes nothing on this set).
4. **llama.cpp on the Pi**: push the GGUF (3.37 GB, held lock window) and
   `build/aarch64/llama-server/llama-server` (sha256 `fc8074d2…`, built by
   `scripts/build_llama_server.sh` into the piquant worktree's `build/`); replay the reference
   prompts with the same payload as `engine_llama.py` (`cache_prompt` on), timing per row and
   VmHWM; compare its probabilities with the x86 reference too. Calibrate first with
   `llama-bench -p 64 -n 0` (expected ~2–3 tok/s if the 0.8B ratio holds).
5. **Clock**: every number above is at 1.5 GHz. If the overclock (2.0 GHz) is applied, re-run
   `cerno-prefill` and set A at the new clock and never mix the two in one table.
6. Prefill tuning for 4B dims (compute-bound): `-b` sweep (8 / 16 / 32 / 64) with
   `pinfer_run_cerno1 --bench`, and `kern_act_group` (16 groups of 2560-column int8 rows = 40 KB,
   over the 32 KB L1D; the kernels agent owns that knob).
7. pinfer KV fp16 (`--kv-dtype fp16`) on set A, to see whether it moves pinfer towards
   llama.cpp's default (f16 KV).
8. The Decision Index `sample/` set (77 rows, references in `cerno-inference/sample/`) needs the
   staged 0.3 suite: 43 datasets, ~7 GB of downloads, HLE gated (the owner's HF login and terms).
   Not done; needs the owner's decision.
9. A synthetic-GGUF unit test for `gguf_to_pq.py` in piquant.

## 9. Timeline (KST)

| time | event | evidence |
|---|---|---|
| 13:48 | piquant worktree reset to main `641a683` | git |
| 13:50–13:58 | cerno-inference cloned and read (all 2101 lines); HF metadata; GGUF header (range request); GGUF downloaded, sha256 = SHA256SUMS | §1 |
| 13:58–14:03 | GGUF vs FP8-checkpoint cross-checks (tiled V heads, untied head, QAT-changed norms) | §2 |
| 14:03 | decision-index kit cloned at `9eb2dbe`; suite staging judged out of scope (§8.8) | kit README |
| 14:05–14:08 | `gguf_to_pq.py` + `--verify`: all equal | §4 |
| 14:07 | llama-server built (host + Pi) | `scripts/build_llama_server.sh` |
| 14:10–14:16 | engine: untied head, `linear_attn.A`, `pinfer_eval_select`; host + Pi builds | piquant `bbcc135` |
| 14:16–14:36 | host gates PASS | piquant journal |
| 14:18–14:21 | reference run on the PC (52 rows); all token ids equal to llama-server's | `results/ref_llama_pc.jsonl` |
| 14:24 | lead: Cerno work moves to this repository | — |
| 14:27–14:29 | `.pq` pushed in a held lock window, md5 checked on the Pi | piquant `logs/harness.jsonl` |
| 14:32 | `cerno-prefill` | §6 |
| 14:39 | `cerno-decide-reuse-a` | §5, §6 |
| 14:45–14:52 | phishing text domain → `.example`, reference re-run (identical numbers for set A); llama.cpp f32-KV / no-FA reference | §5 |
| 14:55 | paused at the lead's request (overclock test); commits | git |
