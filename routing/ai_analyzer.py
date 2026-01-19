from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import re

import torch
from transformers import AutoTokenizer, AutoModel

from .models import AIAnalysis, Evidence, IntentResult, Priority, CallInput
from .nlp_preprocess import build_canonical, PreprocessConfig


def _fmt_ts(seconds: float) -> str:
    s = int(max(0, seconds))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh > 0 else f"{mm:02d}:{ss:02d}"


def _l2_normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(p=2, dim=-1, keepdim=True) + 1e-12)


@torch.inference_mode()
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    s = (last_hidden * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-9)
    return s / d


class AIAnalyzer:
    def analyze(self, call: CallInput, allowed_intents: Dict[str, Dict], groups: Optional[Dict[str, Dict]] = None) -> AIAnalysis:
        raise NotImplementedError


class RubertEmbeddingAnalyzer(AIAnalyzer):
    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        device: Optional[str] = None,
        min_confidence: float = 0.55,
        max_text_chars: int = 4000,
        preprocess_cfg: Optional[PreprocessConfig] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.max_text_chars = max_text_chars

        self.preprocess_cfg = preprocess_cfg or PreprocessConfig(
            drop_fillers=True,
            dedupe=True,
            keep_timestamps=True,
            do_lemmatize=True,
            drop_stopwords=False,
            max_chars=max_text_chars,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._intent_cache_key: Optional[Tuple[str, ...]] = None
        self._intent_mat: Optional[torch.Tensor] = None
        self._intent_ids: List[str] = []

    def analyze(self, call: CallInput, allowed_intents: Dict[str, Dict], groups: Optional[Dict[str, Dict]] = None) -> AIAnalysis:
        prep = build_canonical([(s.start, s.text, s.role) for s in call.segments], self.preprocess_cfg)
        text = prep.canonical_text[: self.max_text_chars]
        lemmas_for_rules = prep.lemmas  # можно использовать дальше (например, для priority/rules)

        intent_ids, intent_emb = self._build_intent_matrix(allowed_intents)

        q = self._embed([text])  # [1, H]
        sims = (q @ intent_emb.T).squeeze(0)  # [N]

        best_idx = int(torch.argmax(sims).item())
        best_intent_id = intent_ids[best_idx]
        best_sim = float(sims[best_idx].item())

        conf = max(0.0, min(1.0, (best_sim - 0.2) / 0.6))

        priority: Priority = "normal"
        if best_intent_id.startswith(("billing.", "access.", "tech.")):
            priority = "high"

        evidence = self._simple_evidence(prep, allowed_intents.get(best_intent_id, {}).get("examples", []))

        suggested_targets = []
        meta = allowed_intents.get(best_intent_id, {})
        gid = meta.get("default_group")
        if gid:
            suggested_targets.append({"type": "group", "id": gid, "confidence": conf})

        notes = f"rubert-embed sim={best_sim:.3f}"
        if conf < self.min_confidence:
            notes += " (low_confidence)"

        intent = IntentResult(intent_id=best_intent_id, confidence=conf, evidence=evidence, notes=notes)
        return AIAnalysis(
            intent=intent,
            priority=priority,
            suggested_targets=suggested_targets,
            raw={"mode": "rubert_embed", "sim": best_sim, "prep_meta": prep.meta, "lemmas_n": len(lemmas_for_rules)},
        )

    def _build_intent_matrix(self, allowed_intents: Dict[str, Dict]) -> Tuple[List[str], torch.Tensor]:
        ids = sorted(allowed_intents.keys())
        key = tuple(ids)
        if self._intent_cache_key == key and self._intent_mat is not None:
            return self._intent_ids, self._intent_mat

        emb_list = []
        self._intent_ids = []
        for intent_id in ids:
            meta = allowed_intents[intent_id]
            examples = meta.get("examples") or [meta.get("title", intent_id)]
            ex_text = " ; ".join(str(x) for x in examples[:10])
            v = self._embed([ex_text])  # [1, H]
            emb_list.append(v)
            self._intent_ids.append(intent_id)

        mat = torch.cat(emb_list, dim=0)  # [N, H]
        self._intent_cache_key = key
        self._intent_mat = mat
        return self._intent_ids, mat

    def _embed(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        return _l2_normalize(pooled)

    def _simple_evidence(self, prep, examples: List[str]) -> List[Evidence]:
        if not examples:
            return []

        # сравниваем по нормализованным словам примеров
        ex_words = set((" ".join(examples)).lower().split())

        scored = []
        # prep.lines содержит строки с таймкодом; выкинем таймкод для сравнения
        for line in prep.lines:
            txt = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", line).strip()
            w = set(txt.split())
            inter = len(w & ex_words)
            if inter > 0:
                scored.append((inter, line))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Evidence] = []
        for _, line in scored[:2]:
            # timestamp можно вытащить из "[MM:SS]"
            m = re.match(r"^\[(\d{2}:\d{2})\]\s*(.*)$", line)
            if m:
                ts, txt = m.group(1), m.group(2)
                out.append(Evidence(text=txt, timestamp=ts))
            else:
                out.append(Evidence(text=line, timestamp="00:00"))
        return out
