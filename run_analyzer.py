import json
from typing import Dict

from routing.models import CallInput, Segment
from routing.ai_analyzer import RubertEmbeddingAnalyzer

def load_intents(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 1. Загружаем JSON
with open("test_data/test_call.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# 2. Преобразуем segments -> dataclass Segment
segments = [
    Segment(
        start=s["start"],
        end=s["end"],
        speaker=s.get("speaker"),
        role=s.get("role"),
        text=s["text"],
    )
    for s in raw["segments"]
]

call = CallInput(
    call_id="test-dengi-001",
    input_audio=raw.get("input"),
    segments=segments,
    meta={
        "mode": raw.get("mode"),
        "role_mapping": raw.get("role_mapping"),
        "note": raw.get("note"),
    },
)
    
allowed_intents = load_intents("configs/intents.json")

# 4. Создаём анализатор
analyzer = RubertEmbeddingAnalyzer()

# 5. Запускаем анализ
result = analyzer.analyze(call, allowed_intents)

# 6. Печатаем результат
from pprint import pprint
pprint(result)
