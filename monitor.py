"""Production-shaped wrapper: score every RAG response, append to a JSONL log, and alert on drift.

Drift rule: rolling mean grounded_score over the last `window` responses drops more than
`max_drop` below the baseline mean (first `window` responses), OR the share of NOT_GROUNDED
verdicts in the window exceeds `max_ungrounded_rate` (PARTIALLY_GROUNDED is not counted as a failure).
"""
from __future__ import annotations
import json, time, collections, pathlib
from dataclasses import asdict
from groundedness import GroundednessScorer

class GroundednessMonitor:
    def __init__(self, scorer: GroundednessScorer, log_path: str = "groundedness_log.jsonl",
                 window: int = 20, max_drop: float = 0.10, max_ungrounded_rate: float = 0.25):
        self.scorer, self.log_path = scorer, pathlib.Path(log_path)
        self.window, self.max_drop, self.max_ungrounded_rate = window, max_drop, max_ungrounded_rate
        self.recent = collections.deque(maxlen=window)
        self.baseline: float | None = None
        self.n = 0

    def observe(self, request_id: str, question: str, answer: str, chunks: list[str]) -> dict:
        r = self.scorer.score(answer, chunks)
        rec = {"ts": time.time(), "request_id": request_id, "question": question,
               "grounded_score": r.grounded_score, "verdict": r.verdict, "is_grounded": r.is_grounded,
               "n_flagged": r.n_tokens_flagged, "n_scored": r.n_tokens_scored,
               "unsupported_spans": [s.text for s in r.unsupported_spans], "latency_ms": r.latency_ms}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.recent.append(r); self.n += 1
        alert = self._check_drift()
        if alert: rec["alert"] = alert
        return rec

    def _check_drift(self) -> str | None:
        if len(self.recent) < self.window: return None
        mean = sum(x.grounded_score for x in self.recent) / len(self.recent)
        rate = sum(x.verdict == "NOT_GROUNDED" for x in self.recent) / len(self.recent)
        if self.baseline is None:
            self.baseline = mean; return None
        if mean < self.baseline - self.max_drop:
            return f"DRIFT: rolling grounded_score {mean:.3f} vs baseline {self.baseline:.3f} (drop > {self.max_drop})"
        if rate > self.max_ungrounded_rate:
            return f"DRIFT: {rate:.0%} of last {self.window} responses NOT_GROUNDED (limit {self.max_ungrounded_rate:.0%})"
        return None


if __name__ == "__main__":
    # Simulated traffic: 20 healthy responses, then a bad deployment that hallucinates.
    import random; random.seed(0)
    from demo import CONTEXT, ANSWERS, QUESTION
    good = [ANSWERS["1. fully supported (near-verbatim)"], ANSWERS["2. paraphrased, still grounded"]]
    bad  = [ANSWERS["3. one fabricated number + entity"], ANSWERS["4. fully hallucinated"]]
    mon = GroundednessMonitor(GroundednessScorer(), log_path="groundedness_log.jsonl", window=10, max_drop=0.10)
    for i in range(30):
        ans = random.choice(good) if i < 15 or random.random() < 0.3 else random.choice(bad)
        rec = mon.observe(f"req-{i:03d}", QUESTION, ans, CONTEXT)
        flag = "  <-- " + rec["alert"] if "alert" in rec else ""
        print(f"{rec['request_id']}  score={rec['grounded_score']:.3f}  {rec['verdict']:19s}{flag}")
    print(f"\nlog written to {mon.log_path.resolve()}  ({mon.n} records)")
