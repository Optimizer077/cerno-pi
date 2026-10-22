"""cerno-pi: Cerno decision models (typed questions -> calibrated probabilities) on pinfer.

    import pinfer
    from cerno_pi import Decider
    with pinfer.Model.load("cerno-4b-q4_0.pq") as m:
        d = Decider(m)                      # temperature from decider_config.json (4B: 1.534)
        r = d.decide("rm -rf /tmp/cache", {"destructive": {"type": "noul",
                     "instructions": "Would this delete data?"}})
        print(r.answers["destructive"])     # {"type": "noul", "noul": 0.97...}
"""
from __future__ import annotations

from .decide import Decider, Decision, Question, label_table, render_question, render_state

__version__ = "0.1.0"

__all__ = ["Decider", "Decision", "Question", "__version__", "label_table", "render_question",
           "render_state"]
