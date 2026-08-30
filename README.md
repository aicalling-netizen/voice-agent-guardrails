# voice-agent-guardrails

Deterministic safety rails for voice agents that run on real phone lines.

An LLM on a phone call has no undo. If it mishears a caller, or confirms a booking
that never actually happened, a real person shows up on the wrong day — and nobody
finds out until they do. You cannot prompt that away reliably: "your appointment is
confirmed" is an extremely plausible thing to say after a booking attempt, whether
or not the booking succeeded.

These are three guards that sit **outside** the model, extracted from running a
production receptionist agent at roughly 400 autonomous calls a day over GSM/SIP
telephony, in Arabic and English.

No dependencies. Pure state machines. Python 3.10+.

```bash
pip install -e ".[dev]"
pytest
```

---

## 1. ActionGuard — never confirm what didn't happen

The agent may only claim an action completed if a tool call actually reported
success. The guard keeps a ledger of attempts and checks the draft utterance
against it before the agent speaks.

```python
from voice_guardrails import ActionGuard

guard = ActionGuard()

with guard.attempt("book_appointment") as attempt:
    result = calendar.create(slot)
    attempt.succeeded(reference=result.id)

verdict = guard.review(draft_reply)
if not verdict.ok:
    draft_reply = verdict.safe_reply     # hand off instead of lying
    log.warning("blocked false confirmation: %s", verdict.reason)
```

Three details that matter more than they look:

- **Pending is not success.** An attempt that neither succeeded nor failed is
  treated as failure. Silence is never read as confirmation.
- **Intent is not a claim.** "I'll book that for you now" passes; "You're all
  booked" after a failed call does not. Mixed utterances are split per sentence,
  so *"Let me try. You're booked."* is still caught.
- **Exceptions mark the attempt failed and re-raise**, so a crash mid-booking
  can't leave the ledger looking clean.

## 2. SilenceWatchdog — close calls that stopped being conversations

Calls don't always end. The caller sets the phone down, the line goes quiet, or
the agent ends up talking at a recording. Each one holds a telephony channel and
a concurrency slot until something upstream times out — and on a gateway with a
fixed port count, a few of those block real callers.

Two clocks, because they catch different failures:

| Clock | Trips when | Why separate |
|---|---|---|
| **silence** | nobody has spoken | Caller may have walked away — nudge, then close politely |
| **no progress** | audio exists but nothing advances | Agent may be talking to a machine — re-prompting only makes more recording |

```python
from voice_guardrails import SilenceWatchdog, WatchdogAction

w = SilenceWatchdog(silence_reprompt_s=6, silence_hangup_s=14)

w.caller_spoke()          # resets both clocks
w.agent_spoke()           # resets silence only — see below
w.progressed()            # a slot filled, a tool ran, intent moved on

d = w.check()
if d.action is WatchdogAction.REPROMPT:
    await say(d.say)
elif d.action is WatchdogAction.HANG_UP:
    await say(d.say); await hangup()
```

`agent_spoke()` deliberately does **not** count as progress. An agent that keeps
talking while nothing comes back is the exact runaway this is built to catch —
if agent audio reset the progress clock, the watchdog would never fire on it.

The clock is injectable (`clock=`). That's not test ceremony: thresholds like
these get tuned under pressure, and a watchdog whose time source you can't
control is one you can't write a regression test for.

## 3. Voicemail detection — stop before the greeting finishes

On outbound dialling a meaningful share of calls reach an answering machine. An
agent that can't tell delivers its whole opening to a recording, waits for a reply
that never comes, and marks the lead "contacted" in the CRM.

Two independent signals, because neither is trustworthy alone:

- **Transcript** — greeting phrasing is formulaic in every language, and the beep
  is often transcribed. Precise, but arrives late (post-STT).
- **Prosody** — a recording is one long unbroken utterance with no turn-taking.
  Earlier and language-independent, but noisy.

```python
from voice_guardrails import classify

v = classify(
    transcript,
    first_utterance_s=7.4,
    speech_ratio=0.93,
    turns_taken=0,
)
if v.should_hang_up:            # STRONG only
    await hangup()
```

`STRONG` requires both signals to agree (or unambiguous transcript evidence:
greeting phrasing *and* a beep). A single signal yields `WEAK` — enough to ask one
short question and see whether a human answers, not enough to hang up on someone
who simply answered the phone with a long sentence. Hanging up on a real caller is
worse than talking to a machine for five seconds, so the asymmetry is deliberate.

English and Gulf-Arabic phrasing ship by default; `_GREETING_PATTERNS` is the part
you'll tune per market.

> One bug worth stealing: `\b` word boundaries don't behave around Arabic script.
> `الصفارة` carries the `ال` prefix, so `\bصفارة\b` never matches — the preceding
> `ل` is itself a word character. The Latin alternatives keep their boundaries; the
> Arabic ones drop them.

---

## Testing

```bash
pytest -q     # 21 tests
```

The suite covers the cases that actually cost money: pending-treated-as-success,
intent-vs-claim, exceptions mid-action, an agent talking to a recording, and a
chatty human who must not be hung up on.

## Scope

Framework-agnostic on purpose — no LiveKit, Pipecat, Twilio or provider SDK
imports. Wire the events in from whatever stack you run.

These are extracted, genericized patterns, rewritten from scratch. No client code,
configuration or data is included.

MIT licensed.
