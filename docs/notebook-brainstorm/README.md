# Notebook brainstorm

A write-up of the planning notebook behind Companion — what's on the pages,
how it compares to what's actually built, and how the ideas link into one
architecture.

- [`ideas.md`](./ideas.md) — the six ideas transcribed from the notebook,
  each with an architecture sketch and a built / not-built check against
  this repo.
- [`mediator-thesis.md`](./mediator-thesis.md) — the corrected framing
  ("digital world" = everything in 1s and 0s; "decision-making" = predicting
  what the user would decide, not the AI's own judgment) and what it changes
  about each idea.
- [`mediator-architecture.html`](./mediator-architecture.html) — a diagram
  of the full loop (open it in a browser). Solid = built in this repo,
  outline = on the page but not built, dashed = blocked on the outlined
  pieces existing first.

## The one-line version

Build a machine that, once it has watched a user long enough to predict
their own decisions accurately, can act on the digital world as their
proxy — and knows, precisely, when it hasn't watched long enough yet.

Roughly half of that is shipped: the belief/confidence math (`belief.py`),
the ask-vs-act gate (`gate.py`), and the memory/retrieval pipeline
(`memory/`, `agents/retriever.py`). The half that isn't: anything that lets
the Agentic AI act on the user's digital world, and the training loop that
would calibrate it against the user's real decisions.
