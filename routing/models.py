from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Priority = Literal["low", "medium", "high", "critical"]

@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    role: Optional[str]
    text: str

@dataclass
class CallInput:
    call_id: str
    input_audio: Optional[str] = None
    segments: List[Segment] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Evidence:
    text: str
    timestamp: str

@dataclass
class IntentResult:
    intent_id: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    notes: str = ""

@dataclass
class AIAnalysis:
    intent: IntentResult
    priority: Priority
    suggested_targets: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingDecision:
    case_id: str
    call_id: str
    intent_id: str
    intent_confidence: float
    priority: Priority
    target_type: Literal["user", "group", "queue", "oncall"]
    target_id: str
    rule_id: str
    evidence: List[Evidence] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TicketDraft:
    title: str
    description: str
    priority: Priority
    tags: List[str]
    assignee_type: str
    assignee_id: str
    links: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TicketCreated:
    external_id: str
    url: str

@dataclass
class NotificationResult:
    sent: bool
    channel: str  # "email" | "chat"
    target: str
    error: str = ""