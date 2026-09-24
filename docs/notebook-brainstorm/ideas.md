# The six ideas

Transcribed from the planning notebook (7 unique pages, several photographed
twice). Each one checked against what's actually in this repo.

## 1. PAA math — Input → Embedding → Trigger

> `Input = x`, `Embedding = T_n`, `Trigger = k(z)`.
> `Environment = Asset + Backend State`.
> `Digital Environment = Digital Asset + Backend Asset + Agentic AI`,
> read by an Actor / Environment / Observer triad.

**Status: built.** `paa_model.py`'s `AssetEncoder` is the embedding, its
`AdaptiveGate` is the trigger. `state.py`'s `ExternalState` (Actor) /
`InternalState` (Observer) split is the same triad. `gate.py` reuses the
`AdaptiveGate` unmodified inside Companion's clarifier.

## 2. Companion Algorithm — compare the two processes

> User process: intake → randomize a prompt from the request set →
> Application drives O.S → feedback to User.
> Agentic AI process: clone the request from User → randomize from the
> request → feedback to Agentic AI.
> "Compare this two processes. Who did it better, who made the better
> data. Store this in a storage."

**Status: not built.** Nothing in this repo runs two competing traces or
scores them. Read correctly (see `mediator-thesis.md`), this isn't "which
independent run is subjectively better" — it's supervised calibration: run
the user's real process and the Agentic AI's predicted process on the same
input, and measure how closely the prediction matches the real outcome.

## 3. Case 1 / Case 2 — the two directions of a turn

> Case 1 (Human → Agentic AI): input (text/video/audio) → accept →
> remember or search the database → ML model → output.
> Case 2 (Agentic AI → Human): output → satisfaction curve → input.
> "Simulate this and see the result."

**Status: Case 1 built, Case 2 partial.** `agents/retriever.py` (semantic +
episodic + domain-knowledge retrieval) is Case 1's "remember or search."
`feedback.py` / `metrics.py` capture signals and counts for Case 2, but
there's no tracked satisfaction curve driving anything back into the
gate, and no simulate-before-acting step.

## 4. Agentic AI as an OS-level automation layer

> "The Agentic AI tries to aid the User... if granted full access to the
> user's digital world it can get better in decision making, making
> better options for the user as well." Diagram: generic asset boxes
> inside an Environment (Gemini or any backend), next to the Agentic AI.
> "Make Companion: learn about the user, understand the user."

**Status: not built — the biggest gap.** Companion today only closes the
conversational loop (`turn()` → reply). There's no tool-execution surface,
no permission model, no way for it to act on anything outside the chat.

## 5. Companion as User ↔ Digital-World ↔ LLM bridge

> `x = digital world`, `y = Agentic AI`, `z = LLM transformer`.
> Standard flow: `User ⟷ digital ⟷ Agentic AI`, with `LLM` underneath.

**Status: partially built.** The User ↔ Agentic-AI ↔ LLM triangle exists
(`agent.py` orchestrating `llm/`). What's missing is an explicit,
structured "digital world" state object — right now `ExternalState.context`
is a loose dict, not a typed, evolvable world-state.

## 6. User / Agentic-AI / LLM as three chained, jointly-trained nets

> `[x1,x2,x3] = [[y1,y2,y3,y4],[y5,y6,y7,y8]]` — a small user-state
> vector mapped into a digital-environment matrix, plus three stacked
> neural-net diagrams: User-net → Agentic-AI-net → LLM-net,
> "build an artificial digital environment, evolving together."

**Status: not built, and in tension with the current design.** You can't
backprop through a hosted Gemini call, so "jointly evolving" can only ever
mean training the first two stages against some reward and treating the
LLM call as frozen. No reward source exists yet (see `mediator-thesis.md`
on why this idea depends on idea 3 first). It also trades away
`belief.py`'s calibrated, auditable confidence for an opaque trained
encoder — a real conflict with the project's own stated design principle
("confidence as evidence, not a magic float"), worth deciding on purpose
rather than drifting into.
