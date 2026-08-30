"""Stop a call that has stopped being a conversation.

The failure this prevents
------------------------
Calls do not always end. The caller puts the phone down without hanging up, the
line drops to silence, or the agent and a hold-music loop talk past each other.
Without a watchdog the session sits open holding a telephony channel, an LLM
context and a concurrency slot until something upstream times out — and on a
gateway with a fixed number of ports, a handful of these will block real callers.

Two clocks, because they mean different things:

* **silence** — nobody has said anything. Prompt once, then close politely.
* **no progress** — there is audio, but nothing is advancing: the same thing is
  being said in a loop, or the agent is talking to a recording. Close sooner;
  re-prompting an answering machine only produces more recording.

The clock is injectable. That is not testing ceremony — a watchdog whose time
source you cannot control is a watchdog you cannot write a regression test for,
and these thresholds are exactly the kind of thing that gets tuned under
pressure and quietly broken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class WatchdogAction(str, Enum):
    CONTINUE = "continue"
    REPROMPT = "reprompt"
    HANG_UP = "hang_up"


@dataclass
class WatchdogDecision:
    action: WatchdogAction
    reason: str = ""
    say: str | None = None


class SilenceWatchdog:
    """Tracks conversational liveness for one call.

    Feed it events as they happen; call :meth:`check` on a timer (every ~1s is
    plenty). Pure state machine — no threads, no I/O.

    Args:
        silence_reprompt_s: quiet time before nudging the caller.
        silence_hangup_s: quiet time before ending the call.
        no_progress_hangup_s: time without meaningful progress before ending.
        max_reprompts: how many nudges before giving up.
        clock: time source, monotonic seconds. Inject a fake in tests.
    """

    def __init__(
        self,
        silence_reprompt_s: float = 6.0,
        silence_hangup_s: float = 14.0,
        no_progress_hangup_s: float = 45.0,
        max_reprompts: int = 2,
        reprompt_line: str = "Sorry, I couldn't hear you there. Are you still on the line?",
        farewell_line: str = "I'll let you go for now. Please call back any time.",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.silence_reprompt_s = silence_reprompt_s
        self.silence_hangup_s = silence_hangup_s
        self.no_progress_hangup_s = no_progress_hangup_s
        self.max_reprompts = max_reprompts
        self.reprompt_line = reprompt_line
        self.farewell_line = farewell_line
        self._clock = clock

        now = clock()
        self._last_caller_audio = now
        self._last_progress = now
        self._reprompts = 0
        self._closed = False

    # -- events -----------------------------------------------------------
    def caller_spoke(self, *, meaningful: bool = True) -> None:
        """Caller audio arrived. `meaningful=False` for noise or backchannel."""
        now = self._clock()
        self._last_caller_audio = now
        if meaningful:
            self._last_progress = now
            self._reprompts = 0

    def agent_spoke(self) -> None:
        """Agent produced audio. Resets silence but NOT progress — an agent
        talking to itself is exactly the runaway case this exists to catch."""
        self._last_caller_audio = self._clock()

    def progressed(self) -> None:
        """Something real happened: a slot filled, a tool called, intent moved on."""
        self._last_progress = self._clock()

    # -- polling ----------------------------------------------------------
    def check(self, now: float | None = None) -> WatchdogDecision:
        now = self._clock() if now is None else now
        if self._closed:
            return WatchdogDecision(WatchdogAction.HANG_UP, "already closed")

        quiet = now - self._last_caller_audio
        stalled = now - self._last_progress

        if stalled >= self.no_progress_hangup_s:
            self._closed = True
            return WatchdogDecision(
                WatchdogAction.HANG_UP,
                f"no progress for {stalled:.0f}s",
                self.farewell_line,
            )

        if quiet >= self.silence_hangup_s or (
            quiet >= self.silence_reprompt_s and self._reprompts >= self.max_reprompts
        ):
            self._closed = True
            return WatchdogDecision(
                WatchdogAction.HANG_UP, f"silent for {quiet:.0f}s", self.farewell_line
            )

        if quiet >= self.silence_reprompt_s:
            self._reprompts += 1
            self._last_caller_audio = now  # restart the silence clock
            return WatchdogDecision(
                WatchdogAction.REPROMPT,
                f"silent for {quiet:.0f}s (nudge {self._reprompts})",
                self.reprompt_line,
            )

        return WatchdogDecision(WatchdogAction.CONTINUE)

    @property
    def closed(self) -> bool:
        return self._closed
