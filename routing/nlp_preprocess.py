from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from razdel import tokenize as razdel_tokenize, sentenize
from natasha import Doc, MorphVocab, Segmenter
from natasha import NewsEmbedding, NewsMorphTagger

_MORPH = MorphVocab()
_SEGMENTER = Segmenter()
_EMBEDDING = NewsEmbedding()
_MORPH_TAGGER = NewsMorphTagger(_EMBEDDING)

STOP_WORDS = {
    "и","а","но","или","да","нет","это","в","на","к","ко","по","за","для","из","у","мы","вы","он","она","они",
    "я","ты","же","бы","ли","то","вот","там","тут","еще","ещё","уже","ну","ок","ладно","понятно","спасибо"
}

FILLER_PATTERNS = [
    r"^\s*(ал(е|ё)|алло)\s*[.!?]?\s*$",
    r"^\s*(да|да-да|угу|ага|мм+|мгм)\s*[.!?]?\s*$",
    r"^\s*(понятно|ясно|окей|хорошо)\s*[.!?]?\s*$",
    r"^\s*(спасибо)\s*[.!?]?\s*$",
]

SPEAKER_PREFIX_RE = re.compile(r"^\s*(speaker\s*\d+\s*:\s*)", re.IGNORECASE)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{8,}\d)")
MONEY_RE = re.compile(
    r"\b(\d{1,3}(?:[ \u00A0]?\d{3})*(?:[.,]\d{1,2})?)\s*"
    r"(руб(лей|ля|\.|)|р\.|₽|usd|eur|дол(ларов|лара|л\.)|евро)\b",
    re.IGNORECASE
)
NUM_RE = re.compile(r"\b\d+\b")

WS_RE = re.compile(r"\s+")
PUNCT_SPACES_RE = re.compile(r"\s+([,.!?;:])")
MULTI_DOTS_RE = re.compile(r"\.{2,}")


@dataclass
class PreprocessConfig:
    drop_fillers: bool = True
    drop_stopwords: bool = False

    dedupe: bool = True
    dedupe_window: int = 2

    max_chars: int = 4000
    keep_timestamps: bool = True
    prefer_role: str = "звонящий"
    min_client_segments: int = 6

    do_tokenize: bool = True
    do_lemmatize: bool = True

    keep_special_tokens: bool = True


@dataclass
class PreprocessResult:
    canonical_text: str
    lines: List[str]
    sentences: List[str]
    tokens: List[str]
    lemmas: List[str]
    meta: Dict[str, Any]


def normalize_text(text: str) -> str:
    t = text.strip()
    t = SPEAKER_PREFIX_RE.sub("", t)
    t = t.replace("Ё", "Е").replace("ё", "е")
    t = t.lower()

    t = EMAIL_RE.sub(" <email> ", t)
    t = PHONE_RE.sub(" <phone> ", t)
    t = MONEY_RE.sub(" <money> ", t)
    t = NUM_RE.sub(" <num> ", t)

    t = MULTI_DOTS_RE.sub(".", t)
    t = WS_RE.sub(" ", t)
    t = PUNCT_SPACES_RE.sub(r"\1", t)

    return t.strip()


def is_filler(text: str) -> bool:
    t = text.strip().lower()
    for pat in FILLER_PATTERNS:
        if re.match(pat, t, flags=re.IGNORECASE):
            return True
    return False


def dedupe_nearby(texts: List[str], window: int = 2) -> List[str]:
    out: List[str] = []
    recent: List[str] = []
    for t in texts:
        if t in recent:
            continue
        out.append(t)
        recent.append(t)
        if len(recent) > window:
            recent.pop(0)
    return out


def _fallback_tokenize(norm_text: str) -> List[str]:
    return re.findall(r"<[a-z_]+>|[a-zа-я0-9]+", norm_text, flags=re.IGNORECASE)


def tokenize_ru(norm_text: str, keep_special_tokens: bool = True) -> List[str]:
    try:
        toks = [t.text for t in razdel_tokenize(norm_text)]
    except Exception:
        toks = _fallback_tokenize(norm_text)

    out: List[str] = []
    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue
        if re.fullmatch(r"\W+", tok):
            continue
        if not keep_special_tokens and tok.startswith("<") and tok.endswith(">"):
            continue
        out.append(tok)
    return out


def lemmatize(norm_text: str, keep_special_tokens: bool = True) -> Tuple[List[str], List[str]]:
    doc = Doc(norm_text)
    doc.segment(_SEGMENTER)
    doc.tag_morph(_MORPH_TAGGER)

    tokens: List[str] = []
    lemmas: List[str] = []
    for token in doc.tokens:
        token.lemmatize(_MORPH)
        tok = token.text
        lem = token.lemma
        if not keep_special_tokens and tok.startswith("<") and tok.endswith(">"):
            continue
        tokens.append(tok)
        lemmas.append(lem)
    return tokens, lemmas


def split_sentences(norm_text: str) -> List[str]:
    try:
        return [s.text.strip() for s in sentenize(norm_text) if s.text.strip()]
    except Exception:
        return [norm_text] if norm_text.strip() else []


def drop_stopwords(items: List[str]) -> List[str]:
    return [x for x in items if x not in STOP_WORDS]


def build_canonical(
    segments: List[Tuple[float, str, Optional[str]]],
    cfg: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    cfg = cfg or PreprocessConfig()

    selected = [(st, tx, rl) for (st, tx, rl) in segments if (rl is None or rl == cfg.prefer_role)]
    if len(selected) < cfg.min_client_segments:
        selected = segments

    lines: List[str] = []
    raw_kept = 0
    raw_dropped = 0

    for start, raw, _role in selected:
        if not raw:
            raw_dropped += 1
            continue
        if cfg.drop_fillers and is_filler(raw):
            raw_dropped += 1
            continue

        norm = normalize_text(raw)
        if not norm:
            raw_dropped += 1
            continue

        if cfg.keep_timestamps:
            mm = int(max(0, start)) // 60
            ss = int(max(0, start)) % 60
            lines.append(f"[{mm:02d}:{ss:02d}] {norm}")
        else:
            lines.append(norm)

        raw_kept += 1

    if cfg.dedupe:
        lines = dedupe_nearby(lines, window=cfg.dedupe_window)

    canonical_text = "\n".join(lines)
    if len(canonical_text) > cfg.max_chars:
        canonical_text = canonical_text[: cfg.max_chars]

    no_ts_text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", canonical_text, flags=re.MULTILINE)
    sentences = split_sentences(no_ts_text)

    tokens: List[str] = []
    lemmas: List[str] = []

    if cfg.do_lemmatize:
        tokens, lemmas = lemmatize(no_ts_text, keep_special_tokens=cfg.keep_special_tokens)
    elif cfg.do_tokenize:
        tokens = tokenize_ru(no_ts_text, keep_special_tokens=cfg.keep_special_tokens)
        lemmas = tokens[:]

    if cfg.drop_stopwords and tokens:
        tokens = drop_stopwords(tokens)
        lemmas = drop_stopwords(lemmas)

    meta: Dict[str, Any] = {
        "raw_kept": raw_kept,
        "raw_dropped": raw_dropped,
        "chars": len(canonical_text),
        "tokens_n": len(tokens),
        "lemmas_n": len(lemmas),
        "sentences_n": len(sentences),
        "keep_timestamps": cfg.keep_timestamps,
        "dedupe": cfg.dedupe,
        "do_lemmatize": cfg.do_lemmatize,
        "drop_stopwords": cfg.drop_stopwords,
    }

    return PreprocessResult(
        canonical_text=canonical_text,
        lines=lines,
        sentences=sentences,
        tokens=tokens,
        lemmas=lemmas,
        meta=meta,
    )
