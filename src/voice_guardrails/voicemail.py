"""Detect an answering machine before the agent starts talking to it.

The failure this prevents
------------------------
On outbound dialling a meaningful share of calls reach voicemail. An agent that
cannot tell will happily deliver its whole opening to a recording, wait for a
reply that never comes, and hold the line until a timeout. At any volume that is
a large amount of wasted telephony spend and blocked concurrency — and the CRM
fills with "contacted" records for people who were never contacted.

Detection uses two independent signals, because neither is reliable alone:

* **Transcript** — greeting phrasing is formulaic across languages, and the beep
  is often transcribed. High precision, but arrives late (post-STT).
* **Prosody** — a recorded greeting is one long uninterrupted utterance with no
  turn-taking. Earlier and language-independent, but noisier.

Either signal alone gives a weak verdict; together they give a strong one. The
caller decides what to do with each — typically hang up on STRONG, and on WEAK
ask one short question and see whether anyone answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Confidence(str, Enum):
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"


@dataclass
class VoicemailVerdict:
    confidence: Confidence
    signals: tuple[str, ...] = ()

    @property
    def should_hang_up(self) -> bool:
        return self.confidence is Confidence.STRONG


# English + common Gulf-Arabic voicemail phrasing. Extend per market — this list
# is the part you will tune most, and it is cheap to tune.
_GREETING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:please\s+)?leave\s+(?:a|your)\s+(?:message|name and number)\b", re.I),
    re.compile(r"\bafter\s+the\s+(?:tone|beep)\b", re.I),
    re.compile(r"\b(?:i'?m|i am|we'?re|we are)\s+(?:not\s+available|unable to take)\b", re.I),
    re.compile(r"\byou'?ve\s+reached\s+the\s+voice\s*mail\b", re.I),
    re.compile(r"\bthe\s+(?:person|number)\s+you\s+(?:have\s+)?(?:dial+ed|called)\b", re.I),
    re.compile(r"\bis\s+(?:not\s+available|switched\s+off|unavailable)\s+(?:right\s+now|at\s+the\s+moment)\b", re.I),
    re.compile(r"\brecord\s+your\s+message\b", re.I),
    # Arabic
    re.compile(r"اترك\s+رسالتك", re.I),
    re.compile(r"بعد\s+(?:الصفارة|النغمة)", re.I),
    re.compile(r"غير\s+متاح", re.I),
    re.compile(r"لا\s+يمكن\s+الوصول", re.I),
)

# NOTE: \b is unreliable around Arabic script. "الصفارة" carries the ال prefix,
# so a bounded \bصفارة\b never matches — the preceding ل is itself a word char.
# Latin terms keep their boundaries; the Arabic alternative drops them.
_BEEP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[?\b(?:beep|tone)\b\]?", re.I),
    re.compile(r"صفارة"),
)


def inspect_transcript(text: str) -> tuple[bool, tuple[str, ...]]:
    """Look for machine-greeting phrasing in what was said."""
    hits: list[str] = []
    for p in _GREETING_PATTERNS:
        if p.search(text):
            hits.append("greeting_phrase")
            break
    for p in _BEEP_PATTERNS:
        if p.search(text):
            hits.append("beep")
            break
    return bool(hits), tuple(hits)


def inspect_prosody(
    first_utterance_s: float,
    speech_ratio: float,
    turns_taken: int,
    *,
    long_utterance_s: float = 6.0,
    dense_speech_ratio: float = 0.85,
) -> tuple[bool, tuple[str, ...]]:
    """Look for the shape of a recording rather than a conversation.

    Args:
        first_utterance_s: length of the far end's first continuous utterance.
        speech_ratio: fraction of the window that was speech (0..1).
        turns_taken: completed turn exchanges so far.
    """
    hits: list[str] = []
    if turns_taken == 0 and first_utterance_s >= long_utterance_s:
        hits.append("long_unbroken_opening")
    if turns_taken == 0 and speech_ratio >= dense_speech_ratio:
        hits.append("no_turn_taking")
    return bool(hits), tuple(hits)


def classify(
    transcript: str = "",
    *,
    first_utterance_s: float = 0.0,
    speech_ratio: float = 0.0,
    turns_taken: int = 0,
) -> VoicemailVerdict:
    """Combine both signals into one verdict.

    STRONG when both agree, or when the transcript alone is unambiguous (a beep
    plus greeting phrasing). WEAK when only one signal fires — enough to probe
    with a short question, not enough to hang up on a real person.
    """
    text_hit, text_signals = inspect_transcript(transcript)
    prosody_hit, prosody_signals = inspect_prosody(first_utterance_s, speech_ratio, turns_taken)
    signals = text_signals + prosody_signals

    if text_hit and prosody_hit:
        return VoicemailVerdict(Confidence.STRONG, signals)
    if len(text_signals) >= 2:  # greeting phrasing *and* a beep
        return VoicemailVerdict(Confidence.STRONG, signals)
    if text_hit or prosody_hit:
        return VoicemailVerdict(Confidence.WEAK, signals)
    return VoicemailVerdict(Confidence.NONE, ())
