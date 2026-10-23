# Cerno 4B on the Raspberry Pi 4 — lab notebook

Owner: Cerno agent, 2026-10-22 (KST). Engine: pinfer of nibblepi (`github.com/Optimizer077/nibblepi`,
formerly `piquant`), branch `worktree-agent-a5c9f7577a224b708`, rebased on main `72284dc`
(release 0.2.0 + docs). The owner asked: can Cerno 4B be set up, tested carefully and optimized
on the Pi 4?

**Answer: yes.** All Pi numbers are at **1.5 GHz**; §6 has the clock of every run.

- **Correctness:** the published Q4_0 GGUF converts losslessly. On 52 questions, pinfer sees the
  same token ids as llama-server on every row. Its probabilities differ from Cerno's llama.cpp
  engine by at most 0.0108 (mean per question 0.0025; mean over the 213 options 0.0013). The top
  answer is the same on 51/52; the other is a 0.497 / 0.503 tie. llama.cpp differs from itself by
  as much: f16 vs f32 KV gives max 0.0092; ARM vs x86 gives max 0.0146.
- **Speed:** a decision (2–5 questions) takes a median of 27.7 s: 14.4 s for the first question,
  5.5 s for each further question. That is **4.5× faster** than llama.cpp's llama-server running
  the reference protocol on the same Pi (median 27.6 vs 112.4 s on the 8 requests both ran).
  Prefill is 6.4–6.6 vs 2.43–2.45 tokens/s.
- **Memory:** 2395 MiB peak RSS vs 3554 MiB for llama-server; the Pi has 3794 MiB.

## 1. Sources, revisions, licence

| Item | Value |
|---|---|
| Inference code | `github.com/apertixsk/cerno-inference` @ `9941548d836ff53b56e326860f94b93adc79bee9` (2 commits, Apache-2.0), cloned to `E:\Personal_tries\third_party\cerno-inference` (outside every repo); vendors decider `Mapika/decider` @ `1c15a48f199adc0eb1952b17441516b5358d79a0` (Apache-2.0) |
| Decision Index kit | `github.com/apolinario/decision-index` @ `9eb2dbe2a358004c8782c66e40a83ac07b953fec` (MIT), cloned to `E:\Personal_tries\third_party\decision-index`; only `decision_index.engines.base` is used (the engine base class) |
| GGUF | `apertixai/cerno-4b-GGUF` @ `61c1bc809b113848001627e6d6ae8bfa1d64daee`, `cerno-4b-v1.1-q4_0.gguf`, 3,373,287,808 B, sha256 `93a94cc5216efd09f6116fbc3f50ec2a621cb7c3bb4b1a8da71734fd6ad25eca` (= the repo's `SHA256SUMS`), md5 `038964105467d52a388591e819d16a67` (checked on the Pi) |
| Tokenizer + config | `apertixai/cerno-4b` @ `ed4b5c59b542ee96c227ee874776c973980c640b`: `tokenizer.json` sha256 `06b95093…a0e523`, `tokenizer_config.json` `171ecbe7…`, `config.json` `35b5c20e…`, `decider_config.json` `bfff899a…` (temperature **1.534**), `chat_template.jinja` `a4aee8af…`. The FP8 weights (`model.safetensors`, 4.84 GB, sha256 `392544403c…`) were **not** downloaded; a few small tensors were read with HTTP range requests for cross-checks |
| Converted model | `cerno-4b-q4_0.pq`, 3,362,381,824 B, md5 `7d0c7ca610490eddfedba6095726744a` (checked on the Pi) + sidecar `cerno-4b-q4_0.tokenizer/` (tokenizer files, `decider_config.json`) |
| Engine builds | `pinfer_run_cerno2` md5 `537fb9a3baa8e949cbe6fe6618bc82c9`, `libpinfer.so` md5 `bc3004daf6b5ae82db88a9a99ca60fa1` (nibblepi branch above, `MATVEC=fast`); the first build `pinfer_run_cerno1` `34f2946a…` (main `641a683` + the same changes) gives bit-identical probabilities (31/31 rows) |
| llama.cpp | v0.7.0 @ `620d222` (nibblepi's baseline), `llama-server` built by `scripts/build_llama_server.sh`: Pi binary sha256 `fc8074d2836b8e39…`, x86 host build for the reference; `llama-bench` of nibblepi's baseline build on the Pi |
| Pi files | `~/pinfer/models/cerno-4b-q4_0.pq` + `.tokenizer/`, `~/pinfer/models/cerno-4b-v1.1-q4_0.gguf`, `~/pinfer/engine/pinfer_run_cerno{1,2}`, `~/pinfer/tmp/cerno/` — listed with undo commands in nibblepi's `docs/pi-state.md` |

**Weights licence.** The model cards of `apertixai/cerno-4b` and `apertixai/cerno-4b-GGUF` declare
`license: apache-2.0` (YAML and a "Licence and credits" section: "Apache-2.0, like the FP8 weights
it was made from (base model Qwen3.5-4B-Base, Apache-2.0)"), but neither weight repository has a
LICENSE file of its own (the FP8 repo's `inference/LICENSE` is the code's). This project uses the
weights only locally, for testing; no weight, GGUF or `.pq` is committed or redistributed.

Third-party code was read in full before anything ran. It ran only from a separate uv venv (Python
3.12, transformers 4.57.6, torch 2.14.1+cpu, tokenizers 0.22.2, decision-index 0.3 from the clone),
with `python -I`, from outside the clone directories.

## 2. The model

From the GGUF metadata and `config.json`:
- Qwen3.5 hybrid, hidden 2560, ffn 9216, vocab 248,320, eps 1e-6, context 262,144;
- 32 layers, with full attention at 3, 7, …, 31 (mask `0x88888888`);
- full attention: 16 query / 4 KV heads × 256, gated, RoPE on 64 dims, θ 1e7;
- linear attention: 16 K heads / 32 V heads × 128, conv 4;
- **no MTP** (no `nextn` block).

Tensors: body Q4_0 2007.6 MB, `token_embd` Q8_0 675.4 MB, `output` Q8_0 675.4 MB, F32 3.8 MB.

What the converter has to respect (nibblepi `engine/FORMAT.md`, "Importing a llama.cpp GGUF"):
- **Untied head.** `config.json` says `tie_word_embeddings: true`, but in the GGUF
  `output.weight` differs from `token_embd.weight` in 111,621 of 248,320 rows (correlation 0.99999).
  The engine gained `lm_head.weight` support.
- **V heads in llama.cpp's tiled order** (`_LinearAttentionVReorderBase`, llama.cpp
  `conversion/qwen.py`). pinfer uses HF's grouped order (`kh = h / rep`), so the converter
  un-tiles. The direction was checked against the HF checkpoint: `dt_bias` is equal in 23 of 24
  layers, and layer 25 differs in one value by 1.1e-4.
- `ssm_a` = −exp(A_log), computed by llama.cpp's converter. It is stored as `linear_attn.A` and
  used directly by the engine.
- The GGUF is **not** the FP8 checkpoint re-quantized. Its norm weights differ from `1 + w` by up
  to 4e-3 and its conv weights by 1e-3. So the reference is llama.cpp on this GGUF.
- Q8_0 codes are in [−127, 127], and all scales are positive and finite.

**Memory budget** (Pi 4: 3794 MiB, 3548 MiB available at idle):
- weights touched per forward: 1915 MiB (body) + the label tiles of the head + a few embedding rows;
- recurrent state: 48 MiB; conv state: 2.3 MiB;
- KV: 64 KiB per token;
- a snapshot: ~51 MiB + KV.

Measured peak RSS (VmHWM, Python process included): **2319.6 MiB** (set A) and **2394.8 MiB**
(set B, longer states) with `eval_select`; 2968.5 MiB when reading the full head; 2627.6 MiB for
the C tool with the full head; llama-server **3514.4 / 3553.9 MiB**.

## 3. The decision protocol (what `cerno_pi.decide` reproduces)

From `engine_llama.py` and `decider/{prompt,systemone,infer}.py` (the vLLM engine uses the same rows
and readout):
- **One row per question** (`plan_rows(rqs, isolated=False)`): `"Context:\n" + state` +
  `"\n\nQuestion: <instructions>\nOptions:\n(A) <opt>\n…\nAnswer: ("`.
  - Up to 10 options are lettered A..J, and the whole row is tokenized as one string (Cerno's
    `JOINT_FIRST_PIECE`).
  - More options get one label token each (A..Z, then the two-letter single tokens; 255 in all).
  - States are cut at 16,384 tokens.
- **JSON states:** `json.dumps(annotate_indices(state), ensure_ascii=False)`.
- **Option rendering:** choice `"name: description"`; score `"i: description"`; yes/no `no` / `yes`.
  "None of the above"-like options are shown as "not listed here".
- **Tokenization in the reference:** the llama.cpp engine decodes the row to text, and
  llama-server tokenizes it again. No BOS is added: the `/tokenize` ids equal the HF tokenizer's
  on all 52 rows.
- **Readout:** the top-260 probabilities of the full-vocabulary softmax; the label entries are
  renormalised after `p^(1/T)`. That equals `softmax(label logits / T)`, except that a label
  outside the top 260 counts as 0. That never happened on this set; the full-head run with
  `top_n=260` gives bit-identical probabilities (31/31).

## 4. Converter exactness (nibblepi `tools/gguf_to_pq.py --verify`)

| Check | Result |
|---|---|
| tensors | 427 (250 quantized, 177 F32), 192 permuted (24 linear layers × 8) |
| quantized weights | 4,840,488,960: every fp16 scale and every code equals the GGUF's after the permutation; dequantized weights equal bit for bit |
| F32 values | 961,536 equal bit for bit |
| reference packers | `tools/quant.py` `pack_q4` / `pack_q8` rebuild all 3,358,474,240 quantized bytes |
| synthetic test (`tests/test_gguf.py`) | GGUFs written the way llama.cpp writes them (12 V heads over 4 K heads, so the permutation is not its own inverse). Checks: `--verify`; llama.cpp's own reorder of the converted weights gives the GGUF back; an F32 model converts to exactly the tensors of a reference `.pq` built from the HF layout; the engine then generates the same tokens (logits 3.5e-7 apart, A vs A_log). A reversed permutation fails it (20 errors) |

## 5. Correctness against the reference (Cerno's llama.cpp engine, same GGUF)

**Reference.** `bench/ref_llama.py` runs `LlamaGGUFEngine` unmodified against llama-server
(x86 host build, `-c 16384 -t 8 --parallel 1`, f16 KV, flash attention auto).
- Request set: `requests/correctness.json` (dataset: README).
- Results: `results/ref_llama_pc.jsonl`. A second run with f32 KV and no flash attention:
  `results/ref_llama_pc_kvf32_nofa.jsonl`.

**pinfer.** `bench/decide_bench.py` on the Pi: `pinfer_run_cerno2`'s library, 4 threads, fp32 KV,
state reuse, `eval_select`. Results: sets A + B, `cerno-decide-reuse-{a2,b2}`.

| comparison | rows | token ids identical | max \|Δp\| | mean \|Δp\| (per question) | argmax agree | file |
|---|---|---|---|---|---|---|
| **pinfer (Pi) vs llama.cpp (x86, default)** | 52 | **52** | **0.0108** | 0.0025 | **51 / 52** | `results/compare_all_reuse2.json` |
| pinfer vs llama.cpp f32 KV, no flash attention (set A) | 31 | 31 | 0.0113 | 0.0033 | 30 / 31 | `results/compare_reuse_a_vs_kvf32.json` |
| llama.cpp (x86) f16 KV vs f32 KV, no FA | 52 | — | 0.0092 | 0.0027 | 52 / 52 | `bench/compare_refs.py` |
| llama.cpp on the Pi (ARM) vs llama.cpp x86 | 26 | — | 0.0146 | 0.0031 | 24 / 26 | `results/speed_vs_llama.json` |
| pinfer (Pi) vs llama.cpp on the Pi | 26 | — | 0.0122 | 0.0027 | 25 / 26 | same |
| pinfer: state reuse vs none | 9 | 9 | **0** (identical) | 0 | 9 / 9 | `bench/compare_runs.py` |
| pinfer: label rows vs full head | 31 | 31 | **0** (identical) | 0 | 31 / 31 | same |
| pinfer: build cerno2 vs cerno1 | 31 | 31 | **0** (identical) | 0 | 31 / 31 | same |

The only argmax disagreement is `sentiment_review.price_complaint`: llama.cpp 0.497 / 0.503,
pinfer 0.508 / 0.492. Over all 213 option probabilities the mean |Δp| is 0.0013 (figure
`docs/figures/agreement*.png`).

**Reading.** The differences exceed the 1e-3 the lead asked to investigate. They are the size of
llama.cpp's own spread between its KV settings and between its ARM and x86 builds, and switching
llama.cpp to f32 KV does not bring it closer to pinfer. Nothing points to a semantic difference:
the token ids are the same and the weights are the same bit for bit. Known numeric differences
remain:
- llama.cpp rounds each activation block's scale to fp16 and uses `1/d` rather than `127/amax`;
  pinfer keeps fp32 scales;
- the summation order;
- the recurrent and attention kernels.

## 6. Pi measurements

Every run went through `harness/pi.py bench`, `valid=true`, throttled 0x0. The governor is
`ondemand`: two runs began at idle (800 / 900 MHz) and ramped to the 1.5 GHz maximum, which they
reported at the end; all others reported 1.5 GHz before and after.

| run (result file in `results/`) | what | result | clock before → after (kHz) |
|---|---|---|---|
| `cerno-prefill_20261022_143214` | cerno1, `--bench`, 64 / 256 tokens | prefill 5.53 / 5.57 tok/s, peak RSS 2627.6 / 2644.6 MB | 1500000 → 1500000 |
| `cerno-decide-reuse-a_20261022_143916` | cerno1, set A (12 requests / 31 questions) | median 32.6 s; probabilities = cerno2's | 1500000 → 1500000 |
| `cerno-gate-spec-exact_20261022_151016` | gate: `spec_exact.sh ./engine/pinfer_run_cerno2` | **58/58 match**, all 58 md5s = `spec-exact-merge6` (0.2.0) | 800000 → 1500000 |
| `cerno-gate-ctx-check_20261022_151146` | gate: `ctx_check.sh` cerno2 vs `pinfer_run_fast` (d48ffda1) | **5/5 identical** | 1500000 → 1500000 |
| `cerno-prefill-sweep_20261022_151543` | cerno2 prefill, 64 tokens, `-b` 64 / 32 / 16 / 8 / 4; fp16 KV; 256 tokens | 6.40 / 6.60 / 5.58 / 5.72 / 6.14 tok/s; fp16 KV 6.22; 256 tokens 6.60 | 900000 → 1500000 |
| `cerno-decide-reuse-a2_20261022_152103` | set A, reuse | median 27.7 s, first q 14.4 s, later q 5.8 s, 6.51 tok/s, 2319.6 MiB | 1500000 → 1500000 |
| `cerno-decide-reuse-b2_20261022_152644` | set B (6 requests / 21 questions), reuse | median 32.0 s, max 89.3 s, later q 5.0 s, 2394.8 MiB | 1500000 → 1500000 |
| `cerno-decide-noreuse_20261022_153323` | support_triage + long_incident without reuse | 38.0 s and 325.3 s (vs 32.5 and 89.3 with reuse) | 1500000 → 1500000 |
| `cerno-decide-fullhead-a_20261022_154007` | set A reading the full head (`--top-n 260`) | median 27.8 s, later q 5.94 s (vs 5.79), **2968.5 MiB** | 1500000 → 1500000 |
| `cerno-llama-bench_20261022_154948` | `llama-bench -p 64,256 -n 0 -r 2` | pp64 2.43 ± 0.02, pp256 2.45 ± 0.00 tok/s | 1500000 → 1500000 |
| `cerno-llama-server-1_20261022_155928` | llama-server, 4 requests / 15 questions | median 121.6 s, every row `cache_n` 0, 3514.4 MiB | 1500000 → 1500000 |
| `cerno-llama-server-2_20261022_160913` | llama-server, 4 requests / 11 questions | median 87.9 s, 3553.9 MiB | 1500000 → 1500000 |

**All 18 requests (pinfer, cerno2, reuse):**
- decision: median 27.7 s, mean 31.4 s, p80 32.5 s, p95 71.8 s, range 13.6–89.3 s;
- first question: median 14.4 s;
- each further question: median 5.5 s (n 34);
- peak RSS: 2394.8 MiB (`bench/summary.py`).

**pinfer vs llama-server on the same 8 requests** (`bench/speed_table.py`,
`results/speed_vs_llama.json`):

| request | questions | pinfer s | llama.cpp s | speed-up | pinfer first / later q s | llama.cpp first / later q s |
|---|---|---|---|---|---|---|
| support_triage | 4 | 32.5 | 103.7 | 3.2× | 13.8 / 6.2 | 34.3 / 23.1 |
| agent_guardrail | 4 | 29.4 | 118.2 | 4.0× | 9.8 / 6.5 | 25.6 / 30.9 |
| sentiment_review | 4 | 28.4 | 130.1 | 4.6× | 10.3 / 6.0 | 31.2 / 33.0 |
| phishing | 3 | 26.8 | 125.0 | 4.7× | 12.1 / 7.3 | 36.9 / 44.0 |
| intent_smart_home | 2 | 14.9 | 51.1 | 3.4× | 11.0 / 3.9 | 30.5 / 20.6 |
| contract_clause | 4 | 37.2 | 244.0 | 6.6× | 24.5 / 4.2 | 67.8 / 58.7 |
| medical_triage | 2 | 20.0 | 69.3 | 3.5× | 14.1 / 5.9 | 38.2 / 31.1 |
| code_review | 3 | 22.6 | 106.5 | 4.7× | 11.8 / 5.4 | 33.0 / 36.7 |
| **total / median** | 26 | **211.9 / 27.6** | **948.0 / 112.4** | **4.5×** | 12.0 / 5.4 | 33.7 / 33.6 |

Why llama.cpp re-reads every row: Cerno's engine sends `cache_prompt: true`, but llama-server only
reuses the cache of this hybrid model from a checkpoint. In v0.7.0 checkpoints are made 4 and
4 + n_ubatch tokens before the end of a prompt (and at chat user-message starts), never where the
state ends. `--checkpoint-min-step` is 8192, and the rows diverge right after the state. Every
row showed `cache_n` 0, so llama.cpp's latency is the full row length at 2.4 tok/s. Its peak RSS
includes the Q4_0 repacking for the ARM kernels.

## 7. Optimizations and their measured effect

| optimization | effect | evidence |
|---|---|---|
| shared-state snapshot (prefill the rows' common prefix once, restore per question) | long_incident (400-token log, 5 questions) **325.3 → 89.3 s (3.6×)**; support_triage (15-token state) 38.0 → 32.5 s; tokens evaluated on the whole set 7313 → 3703; probabilities bit-identical | `cerno-decide-noreuse` vs `-reuse-{a2,b2}` |
| label rows only (`pinfer_eval_select`: the 4-row tiles of the option labels instead of the 248,320-row Q8_0 head) | peak RSS **2968.5 → 2319.6 MiB (−649 MiB)**; later questions 5.94 → 5.79 s median (−0.15 s, −2.5 %); probabilities bit-identical | `cerno-decide-fullhead-a` vs `-reuse-a2` |
| nibblepi 0.2.0 kernels (batched passes of up to 7 vectors; came with the rebase) | prefill 5.53–5.57 → 6.40–6.60 tok/s (+17 %); set A median 32.6 → 27.7 s | `cerno-prefill` vs `-prefill-sweep`, `-reuse-a` vs `-a2` |
| prefill batch size `-b` 4–64 | no gain over the default 64: 5.6–6.6 tok/s in single runs, within run-to-run spread; default kept | `cerno-prefill-sweep` |
| fp16 KV cache | none at these lengths (6.22 vs 6.40 tok/s at 64 tokens); kept fp32 (exact) | same |
| lossless conversion | the published QAT weights are kept bit for bit (no re-quantization loss) | §4 |
| in-engine checkpoint instead of the state blob (nibblepi `pinfer_checkpoint` / `pinfer_rewind`, `Model.checkpoint` / `rewind`; branch `worktree-agent-a4d7c6bb45401045d`, cerno-pi `feat/checkpoint`): the shared prefix is checkpointed inside the context (only the 50 MiB recurrent + conv state is copied; the prefix's K/V rows stay in place) and each later question rewinds to it, instead of `save_state` (4 Python copies of a ~51 MiB + K/V blob) and `load_state` (recurrent state + K/V rows copied back) | PC (host build, bf16 0.8B stand-in): probabilities identical between the two paths, see `CHANGELOG.md`; **Pi (1.5 GHz, 5 requests / 13 questions, one A/B round): mean decision 20.91 → 20.57 s (−1.6 %), median 19.23 → 18.90 s; 13/13 probabilities identical; snapshot median 240 → 20 ms, restore 105 → 20 ms; peak RSS 2308 → 2202 MiB**; nibblepi gates on the same source: spec_exact 58/58, ctx_check 5/5, Cerno logits identical | nibblepi `engine/CONTEXT.md` §10; Pi result files nibblepi `bench/results/ckpt-decide-{state,ckpt}-r1_20261022_22*.json` |

Not done: speculative or MTP drafting does not apply (no generation, and no MTP block). A
smaller head or embedding format would change the published weights, so it was not done.

## 8. Open items

1. The Decision Index `sample/` set (77 rows; references in `cerno-inference/sample/`) needs the
   staged 0.3 suite: 43 datasets, ~7 GB of downloads, HLE gated behind the owner's Hugging Face
   login and terms. Not done; the owner has to decide.
2. llama-server timed on 8 of the 18 requests (26 of 52 questions): about 16 min of Pi time. The
   other 10 would take about 30 min more at 2.4 tok/s with no prefix reuse.
3. ~~At 2.0 GHz (the approved overclock), re-run the Pi tables.~~ 2.0 GHz froze the board (nibblepi
   journal 2026-10-22 18:10); the owner decided to stay at stock 1.5 GHz. Never mix clocks in one table.
4. Pi measurements of the in-engine checkpoint (§7 last row): `bash tmp/ck/decide_ab.sh ckpt|state` on
   the lead's 5-request set (3 rounds each), all 18 requests once per path, `bench/compare_runs.py`
   between the paths (must be 52/52 identical), peak RSS, and nibblepi's `tools/ckpt/ckpt_pi.sh`
   for the C-level checkpoint / rewind / save / load times on a 38-token and a 387-token prefix.

## 9. Timeline (KST)

| time | event | evidence |
|---|---|---|
| 13:48 | nibblepi worktree reset to main `641a683` | git |
| 13:50–13:58 | cerno-inference cloned and read (all 2101 lines); HF metadata; GGUF downloaded, sha256 = SHA256SUMS | §1 |
| 13:58–14:03 | GGUF vs FP8-checkpoint cross-checks (tiled V heads, untied head, QAT-changed norms) | §2 |
| 14:03 | decision-index kit cloned at `9eb2dbe`; suite staging judged out of scope | §8 |
| 14:05–14:08 | `gguf_to_pq.py` + `--verify`: all equal | §4 |
| 14:07 | llama-server built (host + Pi) | `scripts/build_llama_server.sh` |
| 14:10–14:36 | engine: untied head, `linear_attn.A`, `pinfer_eval_select`; host gates PASS | nibblepi journal |
| 14:18–14:21 | reference run on the PC (52 rows) | `results/ref_llama_pc.jsonl` |
| 14:24 | lead: Cerno work moves to this repository | — |
| 14:27–14:39 | `.pq` pushed (held lock window); `cerno-prefill`; `cerno-decide-reuse-a` | §6 |
| 14:45–14:52 | phishing text domain → `.example`, reference re-run; f32-KV / no-FA reference | §5 |
| 14:55 | paused for the overclock test (commit `fccda41`, wip) | git |
| 15:00–15:05 | resumed; rebase on 0.2.0 (`400e264`), records kept from both sides | nibblepi git |
| 15:03–15:21 | host gates on the rebased tree: golden / spec / state PASS, pytest 130, dims PASS, malformed 43/43 | `/root/cerno-logs/gates_2` (WSL) |
| 15:07–15:12 | `pinfer_run_cerno2` pushed; Pi gates spec_exact 58/58, ctx_check 5/5 | §6 |
| 15:13–15:40 | prefill sweep; sets A and B; no-reuse; full head | §6 |
| 15:22 | rebase on `72284dc` (docs only: the code is identical to the gated tree) | nibblepi git |
| 15:41–16:09 | GGUF + llama-server pushed (held lock window); llama-bench; llama-server runs 1 and 2 | §6 |
| 15:45–16:30 | `tests/test_gguf.py` (synthetic GGUFs; mutation check) and the F32 column permutation it found missing; host gates with it | nibblepi |
| 16:10–16:40 | README figures and animation, dataset section, this notebook | `tools/`, `docs/figures/` |
