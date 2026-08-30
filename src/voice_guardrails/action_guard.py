"""Stop a voice agent confirming an action that never actually happened.

The failure this prevents
------------------------
An LLM calls a tool, the tool fails, and the model says "You're all booked in
for Tuesday at three" anyway. On a text interface that is an annoyance you can
scroll back and correct. On a phone call it is a person who shows up on the
wrong day, and nobody finds out until they do.

You cannot prompt this away reliably. The model is producing plausible speech,
and "your appointment is confirmed" is extremely plausible after a booking
attempt. The fix has to be deterministic and sit *outside* the model: the agent
is only permitted to claim success for an action that a tool call actually
returned success for.

Usage
-----
    guard = ActionGuard()

    with guard.attempt("book_appointment") as attempt:
        result = calendar.create(...)
        attempt.succeeded(reference=result.id)

    # Before the agent speaks:
    verdict = guard.review(draft_reply)
    if not verdict.ok:
        draft_reply = verdict.safe_reply

The guard is intentionally boring. It keeps a ledger of what was actually
attempted and what actually succeeded, and it refuses to let the agent assert
completion beyond that ledger.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class Outcome(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Attempt:
    """One attempt at a side-effecting action."""

    action: str
    outcome: Outcome = Outcome.PENDING
    reference: str | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)

    def succeeded(self, reference: str | None = None) -> None:
        self.outcome = Outcome.SUCCEEDED
        self.reference = reference

    def failed(self, error: str | None = None) -> None:
        self.outcome = Outcome.FAILED
        self.error = error


@dataclass
class Verdict:
    ok: bool
    reason: str | None = None
    safe_reply: str | None = None


# Phrases that assert a side effect completed. Deliberately conservative: we
# would rather re-prompt occasionally than let one false confirmation through.
_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:you'?re|you are)\s+(?:all\s+)?(?:booked|confirmed|scheduled|enrolled|registered)\b", re.I),
    re.compile(r"\b(?:i'?ve|i have)\s+(?:just\s+)?(?:booked|confirmed|scheduled|cancel+ed|rescheduled|enrolled|sent|updated)\b", re.I),
    re.compile(r"\b(?:that'?s|that is|it'?s|it is)\s+(?:now\s+)?(?:booked|confirmed|scheduled|cancel+ed|done|sorted)\b", re.I),
    re.compile(r"\byour\s+(?:appointment|booking|reservation|slot|place)\s+(?:is|has been)\s+(?:booked|confirmed|scheduled|cancel+ed|moved)\b", re.I),
    re.compile(r"\b(?:booking|appointment|reservation)\s+confirmed\b", re.I),
    re.compile(r"\ball\s+set\b", re.I),
)

# Said *before* acting, so it must not trip the guard.
_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:i'?ll|i will|let me|i'?m going to|i am going to|shall i|would you like me to)\b", re.I),
    re.compile(r"\b(?:trying|attempting|checking|one moment|bear with me)\b", re.I),
)

DEFAULT_SAFE_REPLY = (
    "I wasn't able to complete that just now. Nothing has been changed on your "
    "account. Let me put you through to a member of the team so they can sort it out."
)


class ActionGuard:
    """Ledger of attempted side effects, plus a check on what the agent may claim.

    Args:
        safe_reply: what the agent says instead of a false confirmation.
        claim_patterns / intent_patterns: override for other languages or domains.
    """

    def __init__(
        self,
        safe_reply: str = DEFAULT_SAFE_REPLY,
        claim_patterns: tuple[re.Pattern[str], ...] = _CLAIM_PATTERNS,
        intent_patterns: tuple[re.Pattern[str], ...] = _INTENT_PATTERNS,
    ) -> None:
        self.safe_reply = safe_reply
        self._claim_patterns = claim_patterns
        self._intent_patterns = intent_patterns
        self.attempts: list[Attempt] = []

    @contextmanager
    def attempt(self, action: str) -> Iterator[Attempt]:
        """Record an attempt. An exception inside the block marks it failed.

        An attempt left PENDING (neither succeeded nor failed, no exception) is
        treated as *not succeeded* — silence is never taken as success.
        """
        record = Attempt(action=action)
        self.attempts.append(record)
        try:
            yield record
        except Exception as exc:  # noqa: BLE001 — deliberately broad; we re-raise
            record.failed(f"{type(exc).__name__}: {exc}")
            raise

    def succeeded(self, action: str) -> bool:
        """True only if some attempt at this action explicitly succeeded."""
        return any(a.action == action and a.outcome is Outcome.SUCCEEDED for a in self.attempts)

    @property
    def any_success(self) -> bool:
        return any(a.outcome is Outcome.SUCCEEDED for a in self.attempts)

    def claims_completion(self, reply: str) -> bool:
        """Does this draft assert that a side effect already happened?"""
        if any(p.search(reply) for p in self._intent_patterns):
            # Mentions intent. Only a claim if it *also* asserts completion in a
            # separate sentence — "I'll try. You're booked." should still trip.
            sentences = re.split(r"(?<=[.!?])\s+", reply)
            return any(
                any(c.search(s) for c in self._claim_patterns)
                and not any(i.search(s) for i in self._intent_patterns)
                for s in sentences
            )
        return any(p.search(reply) for p in self._claim_patterns)

    def review(self, reply: str) -> Verdict:
        """Check a draft utterance against the ledger before the agent speaks."""
        if not self.claims_completion(reply):
            return Verdict(ok=True)
        if self.any_success:
            return Verdict(ok=True)

        failures = [a for a in self.attempts if a.outcome is Outcome.FAILED]
        pending = [a for a in self.attempts if a.outcome is Outcome.PENDING]
        if failures:
            reason = f"claimed completion but {failures[-1].action} failed: {failures[-1].error}"
        elif pending:
            reason = f"claimed completion but {pending[-1].action} never resolved"
        else:
            reason = "claimed completion but no action was ever attempted"

        return Verdict(ok=False, reason=reason, safe_reply=self.safe_reply)

    def reset(self) -> None:
        """Clear the ledger — call between calls, not between turns."""
        self.attempts.clear()
