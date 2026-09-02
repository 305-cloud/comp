"""
gate.py - the Prompt mechanism from the Prompt-Action-Asset (PAA) model,
reused here as-is: an adaptive gate that fires on *deviation from a
learned baseline*, not on raw signal strength.

Where it's used: agents/clarifier.py keeps one AdaptiveGate per user over
the Clarifier's own confidence score. A person who is usually easy to
understand suddenly producing a hard-to-parse turn is itself a signal --
worth a clarifying question even if the absolute confidence score alone
wouldn't have triggered a question.
"""

from __future__ import annotations

from dataclasses import dataclass


_MIN_OBSERVATIONS_BEFORE_FIRING = 2
"""A single prior turn isn't a "pattern" to deviate from -- theta seeded
from one thin greeting (e.g. "hey", often a low score) would otherwise
make the very next confident, on-topic message look like a huge
deviation and force an unwanted ASK, right when the gate has the least
evidence it needs to judge what's actually normal for this person."""


@dataclass
class AdaptiveGate:
    """theta(t) EMA baseline + deviation-triggered Prompt."""

    alpha: float = 0.1
    delta: float = 0.25
    theta: float | None = None
    observations: int = 0

    def step(self, b_t: float) -> tuple[bool, float]:
        if self.theta is None:
            self.theta = b_t
        deviation = abs(b_t - self.theta)
        fires = deviation >= self.delta and self.observations >= _MIN_OBSERVATIONS_BEFORE_FIRING
        # update baseline AFTER computing deviation
        self.theta = self.theta + self.alpha * (b_t - self.theta)
        self.observations += 1
        return fires, deviation
