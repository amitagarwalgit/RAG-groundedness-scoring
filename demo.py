"""POC demo: score four answers against the same retrieved context."""
import sys
from groundedness import GroundednessScorer

CONTEXT = [
    "Acme Corp reported third-quarter revenue of $4.2 billion, up 12% year over year, "
    "driven by strong demand for its cloud services division.",
    "The company's chief executive, Maria Lopez, said the board approved a $500 million "
    "share buyback program and raised full-year guidance.",
    "Acme ended the quarter with 18,400 employees and opened a new data centre in Dublin.",
]
QUESTION = "How did Acme do last quarter?"

ANSWERS = {
    "1. fully supported (near-verbatim)":
        "Acme Corp's third-quarter revenue was $4.2 billion, up 12% year over year, driven by demand for cloud services.",
    "2. paraphrased, still grounded":
        "Acme grew sales by about twelve percent to $4.2B last quarter, mostly thanks to its cloud business, "
        "and the CEO Maria Lopez announced a buyback.",
    "3. one fabricated number + entity":
        "Acme Corp's third-quarter revenue was $6.8 billion, up 12% year over year, and CEO John Carter "
        "announced a $500 million buyback.",
    "4. fully hallucinated":
        "Acme Corp filed for bankruptcy after regulators fined it $2 billion for data breaches in Singapore.",
}

if __name__ == "__main__":
    scorer = GroundednessScorer()
    print(f"model={scorer.model_name} layer={scorer.layer} device={scorer.device}\n")
    print(f"Question: {QUESTION}\nContext: {len(CONTEXT)} retrieved chunks\n")
    for name, ans in ANSWERS.items():
        r = scorer.score(ans, CONTEXT)
        verdict = r.verdict
        print(f"=== {name} ===")
        print(f"answer: {ans}")
        print(f"grounded_score={r.grounded_score:.3f}  delta_precision={r.delta_precision:.2e}  "
              f"flagged={r.n_tokens_flagged}/{r.n_tokens_scored}  -> {verdict}  ({r.latency_ms:.0f} ms)")
        if r.unsupported_spans:
            print("unsupported spans:")
            for s in r.unsupported_spans:
                print(f"   - '{s.text}'  (support {s.support:.2f}; {s.reason})")
        if "--tokens" in sys.argv:
            print("per-token support:", " ".join(f"{t.strip()}:{v:.2f}" for t, v in r.token_support))
        print()
