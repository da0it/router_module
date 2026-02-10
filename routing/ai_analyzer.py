# routing/ai_analyzer.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import time
import logging

import re

import torch
from transformers import AutoTokenizer, AutoModel

from .models import AIAnalysis, Evidence, IntentResult, Priority, CallInput
from .nlp_preprocess import build_canonical, PreprocessConfig

logger = logging.getLogger(__name__)


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

        logger.info(f"Loading RuBERT model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._intent_cache_key: Optional[Tuple[str, ...]] = None
        self._intent_mat: Optional[torch.Tensor] = None
        self._intent_ids: List[str] = []

    def analyze(self, call: CallInput, allowed_intents: Dict[str, Dict], groups: Optional[Dict[str, Dict]] = None) -> AIAnalysis:
        # Засекаем время обработки
        start_time = time.time()
        
        # Препроцессинг текста
        prep = build_canonical([(s.start, s.text, s.role) for s in call.segments], self.preprocess_cfg)
        
        # Обработка длинных диалогов: берем начало + конец
        text = self._extract_text_with_context(prep.canonical_text, self.max_text_chars)
        
        lemmas_for_rules = prep.lemmas

        intent_ids, intent_emb = self._build_intent_matrix(allowed_intents)

        q = self._embed([text])  # [1, H]
        sims = (q @ intent_emb.T).squeeze(0)  # [N]

        # Топ-3 интента для анализа
        top3_indices = torch.topk(sims, k=min(3, len(intent_ids)))[1]
        top3_intents = [
            {"intent": intent_ids[int(i)], "similarity": float(sims[i].item())}
            for i in top3_indices
        ]

        best_idx = int(torch.argmax(sims).item())
        best_intent_id = intent_ids[best_idx]
        best_sim = float(sims[best_idx].item())

        # Улучшенный расчет confidence с учетом margin
        conf = self._calculate_confidence(sims, best_idx)

        # Обработка неопределенных случаев
        if conf < self.min_confidence:
            logger.warning(
                f"Low confidence {conf:.3f} for call {call.call_id}, routing to triage"
            )
            best_intent_id = "misc.triage"
            priority = "normal"
            notes = f"rubert-embed sim={best_sim:.3f} (confidence_too_low → triage)"
        else:
            # Приоритет из метаданных интента
            meta = allowed_intents.get(best_intent_id, {})
            priority: Priority = meta.get("priority", "normal")
            notes = f"rubert-embed sim={best_sim:.3f}"

        evidence = self._semantic_evidence(prep, allowed_intents.get(best_intent_id, {}).get("examples", []))

        suggested_targets = []
        meta = allowed_intents.get(best_intent_id, {})
        gid = meta.get("default_group")
        if gid:
            suggested_targets.append({"type": "group", "id": gid, "confidence": conf})

        # Время обработки
        processing_time_ms = (time.time() - start_time) * 1000

        # Логирование с метаданными
        logger.info(
            "Intent classified",
            extra={
                "call_id": call.call_id,
                "intent": best_intent_id,
                "confidence": round(conf, 3),
                "similarity": round(best_sim, 3),
                "text_length": len(text),
                "processing_time_ms": round(processing_time_ms, 2),
            }
        )

        intent = IntentResult(intent_id=best_intent_id, confidence=conf, evidence=evidence, notes=notes)
        
        return AIAnalysis(
            intent=intent,
            priority=priority,
            suggested_targets=suggested_targets,
            raw={
                "mode": "rubert_embed",
                "model_version": self.model_name,
                "device": self.device,
                "sim": round(best_sim, 4),
                "processing_time_ms": round(processing_time_ms, 2),
                "top3_intents": top3_intents,
                "prep_meta": prep.meta,
                "lemmas_n": len(lemmas_for_rules),
                "text_length": len(text),
            },
        )

    def _extract_text_with_context(self, text: str, max_chars: int) -> str:
        """
        Обработка длинных диалогов: берем начало + конец
        
        Если текст больше max_chars, берем:
        - Первые 60% от max_chars (начало разговора)
        - Последние 40% от max_chars (конец разговора)
        """
        if len(text) <= max_chars:
            return text
        
        # Распределение: 60% начало, 40% конец
        start_chars = int(max_chars * 0.6)
        end_chars = int(max_chars * 0.4)
        
        start_text = text[:start_chars]
        end_text = text[-end_chars:]
        
        # Соединяем с разделителем
        return start_text + "\n[...]\n" + end_text

    def _calculate_confidence(self, sims: torch.Tensor, best_idx: int) -> float:
        """
        Улучшенный расчет confidence с учетом margin между топ-1 и топ-2
        """
        sorted_sims = torch.sort(sims, descending=True)[0]
        best = float(sorted_sims[0].item())
        second = float(sorted_sims[1].item()) if len(sorted_sims) > 1 else 0.0
        
        # Учитываем margin между топ-1 и топ-2
        margin = best - second
        
        # Комбинируем absolute similarity и margin
        conf_sim = (best - 0.2) / 0.6  # абсолютная схожесть
        conf_margin = min(1.0, margin / 0.3)  # относительное превосходство
        
        # Взвешенная комбинация
        conf = 0.7 * conf_sim + 0.3 * conf_margin
        
        return max(0.0, min(1.0, conf))

    def _build_intent_matrix(self, allowed_intents: Dict[str, Dict]) -> Tuple[List[str], torch.Tensor]:
        ids = sorted(allowed_intents.keys())
        key = tuple(ids)
        if self._intent_cache_key == key and self._intent_mat is not None:
            return self._intent_ids, self._intent_mat

        logger.info(f"Building intent matrix for {len(ids)} intents")
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

    def _semantic_evidence(self, prep, examples: List[str], top_k: int = 2) -> List[Evidence]:
        """
        Улучшенный метод извлечения evidence с использованием semantic similarity
        """
        if not examples or not prep.sentences:
            return []
        
        try:
            # Эмбеддинги примеров
            ex_emb = self._embed(examples)  # [N, H]
            
            # Эмбеддинги предложений из диалога
            sent_emb = self._embed(prep.sentences)  # [M, H]
            
            # Считаем similarity
            sims = (sent_emb @ ex_emb.T).max(dim=1)[0]  # [M]
            
            # Топ-K предложений
            top_indices = torch.topk(sims, k=min(top_k, len(prep.sentences)))[1]
            
            evidence = []
            for idx in top_indices:
                sent = prep.sentences[int(idx)]
                # Найти timestamp для этого предложения
                ts = self._find_timestamp_for_sentence(sent, prep.lines)
                evidence.append(Evidence(text=sent, timestamp=ts))
            
            return evidence
        
        except Exception as e:
            logger.warning(f"Semantic evidence extraction failed: {e}, falling back to simple method")
            return self._simple_evidence(prep, examples)

    def _find_timestamp_for_sentence(self, sentence: str, lines: List[str]) -> str:
        """
        Находит timestamp для предложения в списке lines с timestamps
        """
        # Нормализуем sentence для поиска
        sent_norm = sentence.lower().strip()
        
        for line in lines:
            # Извлекаем timestamp из "[MM:SS] text"
            m = re.match(r"^\[(\d{2}:\d{2})\]\s*(.*)$", line)
            if m:
                ts, txt = m.group(1), m.group(2)
                txt_norm = txt.lower().strip()
                
                # Если предложение содержится в строке
                if sent_norm in txt_norm or txt_norm in sent_norm:
                    return ts
        
        return "00:00"

    def _simple_evidence(self, prep, examples: List[str]) -> List[Evidence]:
        """
        Простой fallback метод извлечения evidence (word overlap)
        """
        if not examples:
            return []

        ex_words = set((" ".join(examples)).lower().split())

        scored = []
        for line in prep.lines:
            txt = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", line).strip()
            w = set(txt.split())
            inter = len(w & ex_words)
            if inter > 0:
                scored.append((inter, line))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Evidence] = []
        for _, line in scored[:2]:
            m = re.match(r"^\[(\d{2}:\d{2})\]\s*(.*)$", line)
            if m:
                ts, txt = m.group(1), m.group(2)
                out.append(Evidence(text=txt, timestamp=ts))
            else:
                out.append(Evidence(text=line, timestamp="00:00"))
        return out