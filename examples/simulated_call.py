"""All three guards on one simulated outbound call.

Run:  python examples/simulated_call.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_guardrails import (  # noqa: E402
    ActionGuard,
    SilenceWatchdog,
    WatchdogAction,
    classify,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> float:
        self.t += s
        return self.t


def line(label: str, msg: str) -> None:
    print(f"  {label:<9} {msg}")


def call_reaches_voicemail() -> None:
    print("\nCALL 1 — outbound, reaches an answering machine")
    transcript = "Hi, you've reached the voicemail of. I'm not available right now, please leave a message after the tone"
    v = classify(transcript, first_utterance_s=7.4, speech_ratio=0.94, turns_taken=0)
    line("detect", f"confidence={v.confidence.value} signals={list(v.signals)}")
    line("action", "hang up before delivering the opening" if v.should_hang_up else "probe with one question")


def call_goes_silent() -> None:
    print("\nCALL 2 — connected, then the caller walks away")
    clock = FakeClock()
    w = SilenceWatchdog(silence_reprompt_s=6, silence_hangup_s=14, max_reprompts=2, clock=clock)
    w.caller_spoke()
    line("t=0s", "caller: 'hello?'")

    for step in (7, 7, 7):
        d = w.check(now=clock.advance(step))
        line(f"t={int(clock.t)}s", f"{d.action.value} — {d.reason}")
        if d.action is WatchdogAction.HANG_UP:
            line("say", d.say)
            break


def booking_fails_agent_tries_to_confirm() -> None:
    print("\nCALL 3 — booking fails, model tries to confirm anyway")
    guard = ActionGuard()

    try:
        with guard.attempt("book_appointment") as attempt:
            raise TimeoutError("calendar gateway timeout")
            attempt.succeeded()  # noqa: unreachable — shown for contrast
    except TimeoutError as exc:
        line("tool", f"book_appointment raised {exc}")

    draft = "Perfect — you're all booked in for Tuesday at three. See you then!"
    line("model", draft)

    verdict = guard.review(draft)
    if verdict.ok:
        line("guard", "allowed")
    else:
        line("guard", f"BLOCKED — {verdict.reason}")
        line("say", verdict.safe_reply)


def booking_succeeds() -> None:
    print("\nCALL 4 — booking succeeds, same sentence is allowed through")
    guard = ActionGuard()
    with guard.attempt("book_appointment") as attempt:
        attempt.succeeded(reference="evt_9f2a")
    line("tool", "book_appointment -> evt_9f2a")

    draft = "Perfect — you're all booked in for Tuesday at three."
    line("guard", "allowed" if guard.review(draft).ok else "blocked")
    line("say", draft)


if __name__ == "__main__":
    call_reaches_voicemail()
    call_goes_silent()
    booking_fails_agent_tries_to_confirm()
    booking_succeeds()
    print()
