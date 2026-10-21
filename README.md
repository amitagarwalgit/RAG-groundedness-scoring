# RAG Groundedness Scoring

Token-level groundedness scoring for retrieval-augmented generation. Given a generated answer and the chunks that were retrieved for it, the scorer tells you how well the answer is supported by the context, which exact spans are unsupported, and whether a number in the answer was invented. A drift monitor wraps the scorer for production logging and alerting.

The similarity function is **Delta Score**: Jensen-Shannon divergence between softmax-normalised contextual token embeddings, used with the greedy token-matching scheme of BERTScore. On SummEval it correlates with expert judgements slightly but significantly better than BERTScore cosine on the same backbone.

## How it works

1. The answer (candidate) and the concatenated retrieved chunks (reference) are embedded with `roberta-large`, layer 17.
2. Each token embedding is L2-normalised and passed through a softmax to form a distribution over hidden dimensions.
3. For every answer token, the Jensen-Shannon divergence to every context token is computed and the minimum kept. This is the token's best match anywhere in the context. The mean over content tokens is Delta Score **precision**.
4. Raw divergences are very small, so each minimum is rescaled against the mean divergence of all answer-context pairs, the "unrelated token" baseline. The result is a per-token **support** value in [0, 1]: 1 means an exact match exists, 0 means the token is as far from the context as a random token.
5. **grounded_score** is the mean support over content tokens. Stopwords, punctuation and special tokens are excluded so function words cannot inflate the score.
6. Tokens with support below `token_threshold` are merged into **unsupported spans** with the original text, so a UI can highlight exactly what the context does not back.
7. **Numeric exact-match rule.** Any number in the answer that does not appear literally in the context is forced to zero support. Soft embedding matching cannot distinguish `4.2` from `6.8`, and fabricated figures are the costliest hallucinations, so this is a hard rule rather than a learned signal.

### Verdicts

| verdict | rule |
|---|---|
| `GROUNDED` | `grounded_score >= answer_threshold` (0.65) and no span with support below `hard_floor` (0.4) |
| `PARTIALLY_GROUNDED` | score passes, but at least one span is clearly unsupported, for example a fabricated entity or number |
| `NOT_GROUNDED` | `grounded_score < answer_threshold` |

Read `PARTIALLY_GROUNDED` as "needs review". It fires for genuine fabrications and also for paraphrases the token matcher cannot resolve.

## Quick start

```bash
pip install -r requirements.txt
python demo.py --tokens
python monitor.py
```

The first call pays model download and warm-up. After that an answer of 20 to 30 tokens against a few hundred tokens of context scores in roughly 100 to 350 ms on an Apple M2, and faster on a CUDA GPU.

## Usage

```python
from groundedness import GroundednessScorer

scorer = GroundednessScorer()            # roberta-large, layer 17, auto device
chunks = retriever.search(question)       # list[str]
answer = llm.generate(question, chunks)

result = scorer.score(answer, chunks)
print(result.verdict, result.grounded_score)
for span in result.unsupported_spans:
    print(span.text, span.support, span.reason)
```

`GroundednessResult` fields:

| field | meaning |
|---|---|
| `grounded_score` | mean token support in [0, 1], higher is better |
| `delta_precision` | raw Delta Score precision, lower is better, kept for fidelity with the metric |
| `verdict` | `GROUNDED`, `PARTIALLY_GROUNDED` or `NOT_GROUNDED` |
| `is_grounded` | `verdict == "GROUNDED"` |
| `unsupported_spans` | list of `Span(text, start_token, end_token, support, reason)` |
| `token_support` | `(token, support)` for every scored token |
| `n_tokens_scored`, `n_tokens_flagged` | counts behind the score |
| `latency_ms`, `model` | for logging |

`result.to_json()` serialises everything for a log line or an API response.

### Configuration

```python
GroundednessScorer(
    model_name="roberta-large",   # any encoder loadable with AutoModel
    layer=17,                     # hidden_states index to use
    temperature=1.0,              # softmax temperature; 1.0 reproduces Delta Score
    content_only=True,            # drop stopwords and punctuation from the score
    token_threshold=0.5,          # below this a token is flagged
    hard_floor=0.4,               # a flagged span below this downgrades the verdict
    answer_threshold=0.65,        # below this the answer is NOT_GROUNDED
    numeric_exact_match=True,     # numbers must appear literally in the context
    max_len=512,
)
```

## Production monitoring

`monitor.py` provides `GroundednessMonitor`, which scores each response, appends a JSON record to a log file and returns a drift alert when either of two conditions holds over a rolling window:

- the rolling mean `grounded_score` drops more than `max_drop` (0.10) below the baseline established on the first window, or
- more than `max_ungrounded_rate` (25%) of the window is `NOT_GROUNDED`.

```python
from monitor import GroundednessMonitor
mon = GroundednessMonitor(GroundednessScorer(), log_path="groundedness_log.jsonl", window=50)

rec = mon.observe(request_id, question, answer, chunks)
if "alert" in rec:
    pager.notify(rec["alert"])
```

Running `python monitor.py` simulates fifteen healthy responses followed by a deployment that hallucinates. The monitor stays silent through the healthy period and alerts once the bad responses accumulate.

## Demo

`demo.py` scores four answers to "How did Acme do last quarter?" against the same three retrieved chunks.

| answer | grounded_score | flagged | verdict | unsupported spans |
|---|---|---|---|---|
| near-verbatim restatement | 0.921 | 0 / 17 | GROUNDED | none |
| paraphrase ("twelve percent", "last quarter") | 0.698 | 4 / 20 | PARTIALLY_GROUNDED | twelve, last, grew, mostly |
| one fabricated number and one fabricated name | 0.748 | 4 / 21 | PARTIALLY_GROUNDED | `6.8` (number not in context), `John Carter` |
| fully hallucinated | 0.447 | 6 / 11 | NOT_GROUNDED | bankruptcy after regulators fined, `2`, breaches in Singapore |

The fabricated figure is caught by the numeric rule and the fabricated name by the embedding distance. The fully hallucinated answer still scores above zero because the company name and generic words like "billion" do match the context, which is why the per-span output matters more than the headline number.

## Where this fits

- **Hallucination gate.** Score every RAG response before it reaches the user; block or caveat `NOT_GROUNDED`, route `PARTIALLY_GROUNDED` to review.
- **Regression tests.** Keep a golden set of question, context and answer triples and compare score distributions across model or prompt changes.
- **Observability.** Log `grounded_score` per response alongside latency and token counts; the monitor turns it into an alert.

## Limitations

- Token matching does not know that "twelve" is "12" or that "last quarter" is "third-quarter". Legitimate paraphrases can be flagged.
- Negation and attribution are invisible. "Acme did not grow" scores the same as "Acme grew".
- A substituted entity is caught only when its tokens are far from everything in the context. A similar-sounding name may pass.
- The encoder window is 512 tokens. For longer contexts, score against each chunk separately and take the per-token maximum support.
- Thresholds were set by hand on the demo. Calibrate on a labelled set of grounded and hallucinated answers before relying on the verdicts.

## Roadmap

- Chunk-wise scoring for long contexts with per-token max across chunks
- Number-word normalisation (twelve to 12, 4.2B to 4.2 billion)
- Threshold calibration against a labelled hallucination set
- Batch API and a small HTTP service
- Evaluate alternative encoders (DeBERTa, ModernBERT, embedding models) for the backbone
