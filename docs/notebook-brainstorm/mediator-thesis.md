# The mediator thesis

Two definitions from the notebook that reframe all six ideas at once:

> "Digital world is everything that exists in 1s and 0s."
>
> "Not that it will make decisions — it should be able to make the
> decision the user will make, from a lot of training. The whole idea is
> to build a machine that can act as mediator for the user in the digital
> world, 1s and 0s."

That's a specific, narrow claim, and it's worth keeping precise:

- **Digital world** is scoped by substrate (binary/software state), not by
  how consequential something is. A todo item and a payment are both "1s
  and 0s"; the notebook doesn't distinguish them by stakes, only by what
  they're made of.
- **Decision-making is prediction, not judgment.** The Agentic AI isn't
  supposed to decide what's good for the user by its own values — it's
  supposed to predict, from enough training/evidence, what the user
  themselves would have chosen, and act as their proxy. A mediator, not
  an independent optimizer.

## What this changes about the six ideas

**Idea 1 (PAA math) is the trust engine.** `concentration = α + β` in
`belief.py` isn't just "how sure am I" — it's literally "how much have I
trained on this user's decision function." The gate in `gate.py` is the
mechanism that already answers "have I watched this user long enough to
stand in for them, on this kind of decision, right now."

**Idea 2 (compare the two processes) is the mediator's calibration loop.**
Run the User's real process and the Agentic AI's predicted process on the
same input; the User's outcome is ground truth, the Agentic AI's is a
guess; the comparison score is how close the guess landed. "Who made the
better data" = which traces are worth training on because the prediction
matched.

**Idea 3's Case 2 is the correctness signal for mediation**, not a vague
satisfaction metric — did the mediated action match what the user would
have actually chosen. "Simulate this and see the result" reads literally:
test the predicted decision in a sandbox and check satisfaction before
ever letting the mediator touch anything real.

**Idea 4 (full access, act on behalf of the user) is licensed by
training, not by permission alone.** Access buys more observation, which
raises concentration (idea 1), which is what actually licenses the
mediator to act — the access itself isn't the thing that makes acting
safe.

**Idea 5 (x/y/z bridge) names the three components:** `x` is the
substrate idea 4 operates on, `y` is the mediator itself (ideas 1 + 2
together), `z` is the generative fallback called when nothing learned
covers the situation.

**Idea 6 (jointly-trained stack) is the idealized, currently-unbuildable
version of the whole thesis** — one function learning "what would this
user decide" end to end. It has no valid training signal until idea 3's
satisfaction tracking is real; that dependency isn't visible in the
notebook sketch, only in trying to write down its loss function.

## One thesis, not six ideas

Page 1 is the "knows when" part (confidence). Page 2 is how it gets
trained (calibration). Page 3 is how it perceives and verifies itself
(memory in, satisfaction out). Pages 4/5 are the mission and the
architecture (mediator over the digital world). Page 6 is the idealized,
fully-learned version of the same thing.

Every open gap traces back to one sentence: *"if granted full access to
the user's digital world it can get better in decision making."* The
access half of that sentence (idea 4) has no code yet. The training half
(idea 2 + idea 3's satisfaction loop) is what should be gating it, and
doesn't exist yet either — which is exactly why idea 4 shouldn't be built
before idea 2/3, whatever else gets prioritized.
