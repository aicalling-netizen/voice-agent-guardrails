import pytest

from voice_guardrails.action_guard import ActionGuard, Outcome
from voice_guardrails.silence_watchdog import SilenceWatchdog, WatchdogAction
from voice_guardrails.voicemail import Confidence, classify


# --------------------------------------------------------------------------
# ActionGuard — the false-confirmation case
# --------------------------------------------------------------------------

def test_blocks_confirmation_when_tool_failed():
    guard = ActionGuard()
    with guard.attempt("book_appointment") as a:
        a.failed("calendar 503")

    verdict = guard.review("Great — you're all booked in for Tuesday at three.")
    assert not verdict.ok
    assert "failed" in verdict.reason
    assert verdict.safe_reply


def test_allows_confirmation_when_tool_succeeded():
    guard = ActionGuard()
    with guard.attempt("book_appointment") as a:
        a.succeeded(reference="evt_123")

    assert guard.review("You're all booked in for Tuesday at three.").ok


def test_pending_is_not_success():
    """Silence must never be read as success."""
    guard = ActionGuard()
    with guard.attempt("book_appointment"):
        pass  # neither succeeded() nor failed()

    assert not guard.review("Your appointment is confirmed.").ok


def test_blocks_confirmation_when_nothing_was_attempted():
    guard = ActionGuard()
    assert not guard.review("That's booked for you.").ok


def test_exception_marks_attempt_failed_and_reraises():
    guard = ActionGuard()
    with pytest.raises(RuntimeError):
        with guard.attempt("book_appointment"):
            raise RuntimeError("gateway timeout")

    assert guard.attempts[0].outcome is Outcome.FAILED
    assert not guard.review("You're booked.").ok


@pytest.mark.parametrize(
    "utterance",
    [
        "I'll go ahead and book that for you now.",
        "Let me check the calendar.",
        "One moment while I look at availability.",
        "Would you like me to book Tuesday?",
    ],
)
def test_intent_is_not_a_completion_claim(utterance):
    """Saying what you're about to do must not trip the guard."""
    guard = ActionGuard()
    assert guard.review(utterance).ok


def test_intent_followed_by_claim_still_blocked():
    """'I'll try. You're booked.' must not slip through on the intent clause."""
    guard = ActionGuard()
    with guard.attempt("book_appointment") as a:
        a.failed("no slots")
    assert not guard.review("Let me try that. You're all booked for Tuesday.").ok


def test_per_action_success_is_tracked():
    guard = ActionGuard()
    with guard.attempt("send_sms") as a:
        a.succeeded()
    assert guard.succeeded("send_sms")
    assert not guard.succeeded("book_appointment")


# --------------------------------------------------------------------------
# SilenceWatchdog
# --------------------------------------------------------------------------

class FakeClock:
    """Controllable time source, so watchdog thresholds are testable exactly."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def test_reprompts_then_hangs_up_on_silence():
    clock = FakeClock()
    w = SilenceWatchdog(
        silence_reprompt_s=5, silence_hangup_s=12, max_reprompts=2, clock=clock
    )

    assert w.check(now=clock.advance(1)).action is WatchdogAction.CONTINUE

    d = w.check(now=clock.advance(5))  # t=6, quiet=6
    assert d.action is WatchdogAction.REPROMPT and d.say

    d = w.check(now=clock.advance(6))  # t=12, quiet=6 since the nudge
    assert d.action is WatchdogAction.REPROMPT

    # third silence window exceeds max_reprompts -> close
    assert w.check(now=clock.advance(6)).action is WatchdogAction.HANG_UP
    assert w.closed


def test_caller_speech_resets_silence():
    clock = FakeClock()
    w = SilenceWatchdog(silence_reprompt_s=5, clock=clock)
    clock.advance(4)
    w.caller_spoke()
    assert w.check(now=clock.advance(4)).action is WatchdogAction.CONTINUE


def test_agent_speech_does_not_count_as_progress():
    """An agent talking to a recording must still trip the no-progress clock."""
    clock = FakeClock()
    w = SilenceWatchdog(no_progress_hangup_s=30, clock=clock)
    for _ in range(5):
        clock.advance(6)
        w.agent_spoke()  # keeps silence fresh, must not extend progress
    d = w.check(now=clock.advance(1))
    assert d.action is WatchdogAction.HANG_UP
    assert "no progress" in d.reason


def test_progress_extends_the_call():
    """Isolate the progress clock: silence thresholds set high so only the
    no-progress deadline is under test."""
    clock = FakeClock()
    w = SilenceWatchdog(
        silence_reprompt_s=999, silence_hangup_s=999, no_progress_hangup_s=30, clock=clock
    )
    clock.advance(25)
    w.caller_spoke()  # refreshes both clocks
    w.progressed()
    # 10s later we are 35s from the start but only 10s from the last progress
    assert w.check(now=clock.advance(10)).action is WatchdogAction.CONTINUE


# --------------------------------------------------------------------------
# Voicemail detection
# --------------------------------------------------------------------------

def test_strong_when_transcript_and_prosody_agree():
    v = classify(
        "Hi, I'm not available right now, please leave a message",
        first_utterance_s=7.5,
        speech_ratio=0.93,
        turns_taken=0,
    )
    assert v.confidence is Confidence.STRONG
    assert v.should_hang_up


def test_strong_on_greeting_plus_beep_alone():
    v = classify("please leave a message after the tone [beep]")
    assert v.confidence is Confidence.STRONG


def test_weak_on_prosody_alone():
    v = classify("", first_utterance_s=8.0, speech_ratio=0.95, turns_taken=0)
    assert v.confidence is Confidence.WEAK
    assert not v.should_hang_up


def test_arabic_greeting_detected():
    v = classify("الرقم المطلوب غير متاح حالياً، اترك رسالتك بعد الصفارة")
    assert v.confidence is Confidence.STRONG


def test_real_conversation_is_not_voicemail():
    v = classify(
        "Hello? Yes, speaking.",
        first_utterance_s=1.2,
        speech_ratio=0.3,
        turns_taken=2,
    )
    assert v.confidence is Confidence.NONE
    assert not v.should_hang_up


def test_long_greeting_by_a_human_is_only_weak():
    """A chatty human must not be hung up on."""
    v = classify(
        "Hello, sorry, I was just getting to the phone, how can I help you today?",
        first_utterance_s=7.0,
        speech_ratio=0.9,
        turns_taken=0,
    )
    assert v.confidence is Confidence.WEAK
    assert not v.should_hang_up
