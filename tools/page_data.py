"""Traceable inputs for the cerno-pi landing README (no new Pi runs)."""
import json
from pathlib import Path
from animation_measurements import (
    DECISION_CPU_S, DECISION_GPU_S, PREFILL, GPU_BO_MIB, GPU_CAP_MIB, GPU_SHARE,
)

ROOT = Path(__file__).resolve().parents[1]
REQUEST = next(r for r in json.loads((ROOT/'requests/correctness.json').read_text(encoding='utf-8'))
               if r['name']=='support_triage')
# Public measured option probabilities, unchanged by the in-engine checkpoint (t006,13/13 exact).
SUPPORT = next(r for r in json.loads((ROOT/'results/cerno-decide-reuse-a2_20261022_152103.json').read_text(encoding='utf-8'))['results']
               if r.get('name')=='support_triage')
# Latest checkpoint timing: nibblepi bench/results/ckpt-decide-ckpt-r1_20261022_222330.json,
# support_triage. First answer is the sum of the three recorded phases, load excluded.
PREFIX_S, CHECKPOINT_S, FIRST_QUESTION_S, REQUEST_S = 2.2333, 0.0635, 10.0151, 29.9716
FIRST_ANSWER_S = PREFIX_S + CHECKPOINT_S + FIRST_QUESTION_S
# Earlier apples-to-apples8-request/26-question comparison, not the later5-request mean.
MATCHED = json.loads((ROOT/'results/speed_vs_llama.json').read_text(encoding='utf-8'))['summary']
SPEEDUP = MATCHED['speedup_total']
# Latest CPU process HWM: cerno-gpu-eng-decide-000-r{1,2,3} (see animation_measurements.py).
# Llama process HWM: results/cerno-llama-server-2_20261022_160913.json / CERNO.md §6.
PINFER_RSS_MIB, LLAMA_RSS_MIB = 2200.2, 3553.9
# results/cerno-llama-bench_20261022_154948.json / CERNO.md §6 (P64 and P256).
LLAMA_PREFILL = {64: 2.43, 256: 2.45}
# Public CPU/reference comparison,52 questions/213 options, published in this repo.
AGREEMENT = json.loads((ROOT/'results/compare_all_reuse2.json').read_text(encoding='utf-8'))['summary']
