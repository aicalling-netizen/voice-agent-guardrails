"""Deterministic safety rails for production voice agents."""

from .action_guard import ActionGuard, Attempt, Outcome, Verdict
from .silence_watchdog import SilenceWatchdog, WatchdogAction, WatchdogDecision
from .voicemail import Confidence, VoicemailVerdict, classify

__all__ = [
    "ActionGuard", "Attempt", "Outcome", "Verdict",
    "SilenceWatchdog", "WatchdogAction", "WatchdogDecision",
    "Confidence", "VoicemailVerdict", "classify",
]
__version__ = "0.1.0"
