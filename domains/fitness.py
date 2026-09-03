"""
fitness.py - a fitness/health coaching domain: same machinery as
general.py and study.py, just pointed at a different purpose and a
different set of required_slots + curated domain_knowledge.
"""

from __future__ import annotations

from domain import DomainConfig

_CURATED_KNOWLEDGE = [
    "A sustainable weight-loss rate is about 0.5kg per week -- faster loss "
    "tends to cost muscle, not just fat, and is harder to keep off.",
    "Protein at every meal helps preserve muscle during a calorie deficit "
    "and keeps hunger down better than cutting fat or carbs alone.",
    "Daily step count is one of the most consistent predictors of "
    "successful long-term weight loss, more than workout intensity alone.",
    "Consistency over weeks matters far more than any single day's "
    "workout or meal -- a missed day is not a failed plan.",
]

FITNESS_DOMAIN = DomainConfig(
    name="fitness_companion",
    purpose=(
        "Be a personal fitness and health companion: track this specific "
        "person's goals, starting point, and progress over time, and give "
        "advice grounded in what's actually true about them -- not generic, "
        "one-size-fits-all fitness content."
    ),
    required_slots=["goal"],
    clarifying_question_bank=[
        "What's your main goal -- losing weight, building strength, more energy, or something else?",
        "What does a typical day look like for you right now in terms of activity and meals?",
    ],
    system_prompt=(
        "You are a personal fitness and health companion. Ground advice in "
        "what you actually know about this user -- their goal, starting "
        "weight, activity level, any injuries or limitations they've "
        "mentioned -- rather than generic fitness content. Favor sustainable, "
        "evidence-based habits over crash diets or extreme plans. Never give "
        "specific medical advice or diagnose anything; suggest a doctor or "
        "registered dietitian for medical concerns."
    ),
    domain_knowledge=_CURATED_KNOWLEDGE,
)
