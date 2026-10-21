"""RAG groundedness scoring with Delta Score.

Candidate  = generated answer
Reference  = retrieved context chunks (concatenated)

For every answer token we compute the Delta Score divergence (Jensen-Shannon over
softmax of L2-normalised contextual embeddings) to every context token and keep the
minimum, i.e. the best match anywhere in the context.  Delta Score precision is the
weighted mean of those minima (lower = better grounded).

Raw Delta divergences live in a very compressed range (~1e-5), so for dashboards we also
report a calibrated per-token support score in [0, 1]:

    support_j = 1 - min_i D_ij / D_random

where D_random is the mean divergence between arbitrary answer/context token pairs,
i.e. "how far is this token from its best match, relative to an unrelated token".
Tokens whose support falls below `token_threshold` are flagged; consecutive flagged
tokens are merged into spans so you can show the user exactly what is unsupported.
"""
from __future__ import annotations
import json, math, re, string, time
from dataclasses import dataclass, asdict, field
from typing import Iterable
import torch
from transformers import AutoTokenizer, AutoModel

STOPWORDS = set("""a an the and or but if of at by for with about as into like through after over
between out against during without before under around among to from in on is are was were be been
being am do does did doing have has had having it its this that these those i you he she we they me
him her us them my your his our their mine yours hers ours theirs what which who whom whose where when
why how all any both each few more most other some such no nor not only own same so than too very s t
can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn
mustn needn shan shouldn wasn weren won wouldn there here then also would could may might must shall
""".split())


def _entropy2(p):  # bits, along last dim
    return -(p * torch.log2(p.clamp_min(1e-300))).sum(-1)


def js_matrix(Pc, Pr):
    """Pc (n,d), Pr (m,d) distributions -> (n,m) Jensen-Shannon divergence in bits, in [0,1]."""
    M = 0.5 * (Pc[:, None, :] + Pr[None, :, :])
    return _entropy2(M) - 0.5 * (_entropy2(Pc)[:, None] + _entropy2(Pr)[None, :])


@dataclass
class Span:
    text: str
    start_token: int
    end_token: int
    support: float          # mean support of the tokens in the span (0 = unrelated, 1 = exact)
    reason: str = "no close token in context"


@dataclass
class GroundednessResult:
    grounded_score: float   # calibrated, in [0,1]; weighted mean token support
    delta_precision: float  # raw Delta Score precision (lower = better); kept for fidelity
    verdict: str            # GROUNDED | PARTIALLY_GROUNDED | NOT_GROUNDED
    is_grounded: bool       # verdict == GROUNDED
    n_tokens_scored: int
    n_tokens_flagged: int
    unsupported_spans: list[Span]
    token_support: list[tuple[str, float]]   # (token text, support) for every scored token
    latency_ms: float
    model: str

    def to_json(self) -> str:
        d = asdict(self); return json.dumps(d, ensure_ascii=False)


class GroundednessScorer:
    def __init__(self, model_name: str = "roberta-large", layer: int = 17,
                 device: str | None = None, temperature: float = 1.0,
                 content_only: bool = True, token_threshold: float = 0.5, hard_floor: float = 0.4,
                 answer_threshold: float = 0.65, numeric_exact_match: bool = True, max_len: int = 512):
        """
        token_threshold     : token support below this is flagged as unsupported
        hard_floor          : a flagged span below this makes an otherwise-passing answer PARTIALLY_GROUNDED
        answer_threshold    : mean support below this is NOT_GROUNDED
        numeric_exact_match : numbers in the answer must appear literally in the context (soft token
                              matching cannot distinguish 4.2 from 6.8, so this closes that gap)
        """
        self.model_name, self.layer, self.T = model_name, layer, temperature
        self.content_only, self.token_threshold, self.hard_floor = content_only, token_threshold, hard_floor
        self.answer_threshold, self.numeric_exact_match = answer_threshold, numeric_exact_match
        self.max_len = max_len
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32).eval().to(self.device)
        self.special = {self.tok.cls_token_id, self.tok.sep_token_id, self.tok.bos_token_id,
                        self.tok.eos_token_id, self.tok.pad_token_id} - {None}

    # ---------- encoding ----------
    @torch.no_grad()
    def _encode(self, text: str):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = enc.to(self.device)
        hs = self.model(**enc, output_hidden_states=True).hidden_states
        H = hs[self.layer][0].float().cpu().double()
        ids = enc["input_ids"][0].cpu().tolist()
        toks = self.tok.convert_ids_to_tokens(ids)
        return H, ids, toks, offsets

    _NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

    @classmethod
    def _numbers(cls, text: str) -> set[str]:
        return {m.replace(",", "") for m in cls._NUM.findall(text)}

    def _missing_numbers(self, answer: str, context: str):
        """(char_start, char_end, number) for every number in the answer absent from the context."""
        ctx = self._numbers(context)
        return [(m.start(), m.end(), m.group().replace(",", "")) for m in self._NUM.finditer(answer)
                if m.group().replace(",", "") not in ctx]

    def _dist(self, H):
        H = H / H.norm(dim=-1, keepdim=True)
        return torch.softmax(H / self.T, dim=-1)

    def _is_content(self, tok_id: int, tok_text: str) -> bool:
        if tok_id in self.special: return False
        w = tok_text.replace("Ġ", "").replace("▁", "").strip().lower()
        if not w or all(ch in string.punctuation for ch in w): return False
        if self.content_only and w in STOPWORDS: return False
        return True

    # ---------- scoring ----------
    def score(self, answer: str, context: str | Iterable[str]) -> GroundednessResult:
        t0 = time.time()
        if not isinstance(context, str):
            context = "\n".join(context)
        Ha, ida, toka, offa = self._encode(answer)
        Hc, idc, tokc, _ = self._encode(context)
        Pa, Pc = self._dist(Ha), self._dist(Hc)
        D = js_matrix(Pa, Pc)                                   # (n_answer, n_context)
        ctx_keep = torch.tensor([i not in self.special for i in idc])
        D = D[:, ctx_keep]                                      # never "match" to CLS/SEP of the context
        min_d = D.min(dim=1).values                             # best match per answer token
        d_random = D.mean().item()                              # unrelated-pair baseline for calibration
        support = (1.0 - min_d / d_random).clamp(0.0, 1.0)

        # numeric exact-match rule: numbers absent from the context get zero support
        forced: dict[int, str] = {}
        if self.numeric_exact_match:
            for cs, ce, num in self._missing_numbers(answer, context):
                for k, (ts, te) in enumerate(offa):
                    if te > cs and ts < ce and ida[k] not in self.special:
                        support[k] = 0.0; forced[k] = f"number '{num}' not in context"

        keep = [self._is_content(i, t) for i, t in zip(ida, toka)]
        idx = [k for k, v in enumerate(keep) if v]
        if not idx:                                             # answer had no content tokens
            return GroundednessResult(1.0, 0.0, "GROUNDED", True, 0, 0, [], [], (time.time()-t0)*1000, self.model_name)

        sup_kept = support[idx]
        min_kept = min_d[idx]
        grounded = float(sup_kept.mean())
        delta_P = float(min_kept.mean())
        flagged = [k for k, s in zip(idx, sup_kept.tolist()) if s < self.token_threshold]

        # merge consecutive flagged tokens (allowing a gap of one non-content token) into spans
        spans: list[Span] = []
        for k in flagged:
            if spans and k - spans[-1].end_token <= 2:
                spans[-1].end_token = k
            else:
                spans.append(Span("", k, k, 0.0))
        for s in spans:
            s.text = self.tok.convert_tokens_to_string(toka[s.start_token:s.end_token + 1]).strip()
            s.support = float(support[s.start_token:s.end_token + 1].mean())
            reasons = {forced[k] for k in range(s.start_token, s.end_token + 1) if k in forced}
            if reasons: s.reason = "; ".join(sorted(reasons))

        if grounded < self.answer_threshold:
            verdict = "NOT_GROUNDED"
        elif any(sp.support < self.hard_floor for sp in spans):
            verdict = "PARTIALLY_GROUNDED"
        else:
            verdict = "GROUNDED"

        token_support = [(toka[k].replace("Ġ", " "), round(float(support[k]), 3)) for k in idx]
        return GroundednessResult(
            grounded_score=round(grounded, 4), delta_precision=delta_P, verdict=verdict, is_grounded=verdict == "GROUNDED",
            n_tokens_scored=len(idx), n_tokens_flagged=len(flagged), unsupported_spans=spans,
            token_support=token_support, latency_ms=round((time.time()-t0)*1000, 1), model=self.model_name)
