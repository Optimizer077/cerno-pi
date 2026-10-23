"""Measured Cerno 4B animation inputs; no model, prompt text or timings are invented.

Raw t006/t007 files live in nibblepi's bench/results (Cerno raw files stay local).
The request text/options are already in this repository's requests/correctness.json.
"""

NAME = "intent_smart_home"
# Every t006/t007 result's state_before.freq_khz is 1500000 on all four cores.
CLOCK_GHZ = 1.5
# results/ref_llama_pc.jsonl temperature field; Cerno 4B's calibration protocol.
TEMPERATURE = 1.534

# ckpt-decide-ckpt-r1_20261022_222330.json, request intent_smart_home.
FLOW = dict(prefix_tokens=26, prefix_s=3.6741, checkpoint_s=0.0195, total_s=13.7052,
            rows=[dict(qid="primary_intent", tokens=46, seconds=6.5175, rewind_s=0.0),
                  dict(qid="multiple_intents", tokens=24, seconds=3.4753, rewind_s=0.0195)])
# Same request/file: unrounded probabilities (temperature 1.534).
PROBS = {"primary_intent": [0.8595535220681793, 0.10811391440172238,
         0.004232648655994877, 0.0038604848722386836, 0.010655098089656341,
         0.013584331912208354], "multiple_intents": [0.07347064166792334, 0.9265293583320767]}

# Paired checkpoint A/B, same five requests: ckpt-decide-state-r1_20261022_222139.json
# and ckpt-decide-ckpt-r1_20261022_222330.json. Medians, not this sample's total.
COPY_SNAPSHOT_MS, CHECKPOINT_MS = 239.886, 19.559174000050916
COPY_RESTORE_MS, REWIND_MS = 105.499, 19.507321000219235
# engine/CONTEXT.md §10a; slot also reported in gpu-eng-final-positive_20261023_143058.json.
CHECKPOINT_BYTES = 52690944  # 52.69 decimal MB; copies linear state, leaves prefix KV in place.

CPU_SOURCES = ["cerno-gpu-eng-decide-000-r1_20261023_134643.json",
               "cerno-gpu-eng-decide-000-r2_20261023_140307.json",
               "cerno-gpu-eng-decide-000-r3_20261023_142609.json"]
GPU_SOURCES = ["cerno-gpu-eng-decide-033-r1_20261023_135004.json",
               "cerno-gpu-eng-decide-033-r2_20261023_142239.json",
               "cerno-gpu-eng-decide-033-r3_20261023_143037.json"]
# Arithmetic means of intent_smart_home's recorded per-request/per-row seconds in those files.
# Row seconds include rewind; total includes checkpoint and Python/request overhead.
CPU = dict(prefix_s=3.710566666666667, checkpoint_s=0.019566666666666666,
           questions=[6.5553, 3.5523], total_s=13.856633333333333)
GPU = dict(prefix_s=3.0755333333333335, checkpoint_s=0.0195,
           questions=[5.4295, 2.900733333333333], total_s=11.444466666666667)
# The same three paired five-request benchmarks, mean seconds per decision.
DECISION_CPU_S, DECISION_GPU_S = 20.66855582313292, 17.264531743132103

# t007-summary.json, prefill_runs/averages: three rounds of each N, flag off / .33,
# cerno-gpu-eng-prefill-{000,033}-p{64,256}-r{1,2,3}_*.json (stock Pi 4, 1.5 GHz).
PREFILL = {64: dict(cpu_s=9.465552037802285, gpu_s=7.830693779268764,
                   cpu_tps=6.762433333333334, gpu_tps=8.1732),
           256: dict(cpu_s=36.99620744152795, gpu_s=31.22080376399116,
                     cpu_tps=6.919866666666667, gpu_tps=8.1999)}
# GPU_SETUP + DRM fdinfo in GPU_SOURCES; gpu-eng-health_20261023_143059.json.
GPU_SHARE, GPU_BO_MIB, GPU_CAP_MIB = 0.33, 498.78, 512
GPU_MATRICES, ELIGIBLE_MATRICES = 138, 200
CPU_CORES, QPUS = 4, 8  # Pi 4 Cortex-A72 cores / VideoCore VI QPUs; t005/t007 CSD=8 QPUs.
# cerno-gpu-eng-agreement-033_20261023_135935.json: 52 questions, 213 option probabilities.
MAX_DP, TOP_MATCHES, N_QUESTIONS = 0.012748099360329168, 50, 52
