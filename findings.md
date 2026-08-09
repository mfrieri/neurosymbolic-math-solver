# Findings

Running log of conclusions and insights worth keeping — things that were not
obvious in advance, cost time to discover, or are worth saying out loud in a
write-up or interview.

**Add to this file whenever a new insight comes up.** Anything that changed how
the system was built, explained a surprising number, or would be easy to forget
and rediscover the hard way belongs here. Each entry: what was observed, why it
happens, what to do about it.

---

## 1. The expected-value objective is under-constrained (the big one)

**Observed.** After training the pipeline with `loss = |predicted_sum − true_sum|`,
sum accuracy was 97.4% but digit accuracy was only 89.1% — and digit 4 scored
exactly **0.0000**. The CNN never once predicted a 4.

Inspecting the softmax over images of a true 4:

```
P(3) = 0.498    P(4) = 0.000    P(5) = 0.453     expected value 4.088, argmax 3
```

**Why.** The predicted sum is an *expected value*, so the loss constrains only
the **mean** of each digit distribution, never its shape. Any distribution
averaging to 4 is equally optimal — one-hot on 4, or half on 3 and half on 5.
The network found the one-hot solution for nine digits and fell into the hedge
for the tenth, with no gradient pressure to leave it, because the hedge is
*exactly as good* under that objective.

**Fix.** Compute the full distribution over the 19 possible sums and apply NLL,
as DeepProbLog does:

```
P(sum = s) = Σ_{i+j=s} P(d₁=i) · P(d₂=j)
```

A hedge now smears probability across neighbouring sums instead of
concentrating it on one, so the objective punishes it. Half-3/half-5 against a
certain 5 puts mass on sums 8 and 10 and **zero on 9** — while its expected
value is a perfectly innocent 9.0.

**Result.** Digit accuracy 0.8909 → **0.9882**. P(4) on images of 4 went
0.000 → **0.986**. Same architecture, same data, same hyperparameters.

Both objectives are kept in the repo (`loss_mode="expected"` / `"distribution"`)
so the comparison is reproducible.

---

## 2. Sum-only supervision matches full digit supervision

| Model | Digit accuracy |
|---|---|
| Sum-supervised, NLL objective | 0.9882 |
| Digit-supervised (Tier 0 baseline) | 0.9898 |
| Gap | **0.0016** |

16 images out of 10,000 — inside run-to-run noise. A CNN that never saw a
single digit label matched one trained on 60,000 of them. This is the project's
central claim.

## 3. Fixing the hedge also improved sum accuracy (0.9744 → 0.9806)

Not obvious, since the L1 model was already near-optimal at its own objective.
Hedged digits **compound**: a 4 spread across 3 and 5, paired with a second
slightly-uncertain digit, produces an expected value that rounds the wrong way
more often. Peaked distributions are more robust under composition.

## 4. Every error is a perception error, never a reasoning error

Example from the trained model: `8+1`, true sum 9, predicted **2.00** — not 8.5,
not something uncertain. Exactly 2.00. The CNN read the handwritten 8 as a 1
with near-total confidence, and the symbolic layer did flawless arithmetic on a
wrong input.

This is the architecture working as designed. Errors are attributable to a
specific half of the system, which an end-to-end network cannot offer.

## 5. Error distribution is bimodal, not diffuse

Across the 10,000-pair test set: **mean** absolute error 0.1029, **median**
0.0000. More than half the predictions are exact to four decimal places while
the mean sits ten times higher.

Errors are rare and large, not spread thinly across all predictions. That is the
signature of a CNN that genuinely learned digits rather than one hedging its way
to decent averages — worth checking, since the two look identical on a mean.

---

## 5b. Compositional generalization: the headline result

Trained on sums 0–9, tested on sums 10–18 (never seen together).

| Model | Held-out accuracy | Params |
|---|---|---|
| Neurosymbolic (sum distribution) | **0.5868** | 108,618 |
| Neurosymbolic (expected value) | 0.3175 | 108,618 |
| End-to-end regression | 0.0720 | 250,817 |

The baseline loses while holding **more than twice the capacity** — this result
is about structure, not size.

**The baseline's curve is the most telling line on the chart.** Its training loss
fell steadily (1.3154 → 0.4382) while held-out accuracy *rose to 0.1578 at epoch
2 and then declined to 0.0720*. Getting better at sums 0–9 made it worse at
10–18: it learned the training range's ceiling as a fact about the world. The
neurosymbolic curve jumps at epoch 3 and stays flat — no learnable parameter
sits on the sums, so there is nothing there to overfit.

## 5c. That 58.7% is one broken digit, not weak generalization

Per-digit accuracy of the part (b) model: digits 0–8 at **0.976–0.997**, digit 9
at **0.0000** (read as 7, with P(7)=0.942).

That single failure accounts for the entire number:

```
test pairs containing a 9:                       39.95%
predicted ceiling if 9 fails, others ~98.8%:     0.5862
observed:                                        0.5868
```

**Accuracy on held-out pairs containing no 9: 0.9770** (n=6005) — sums like
8+8=16 that never appeared in training, computed correctly. Against 0.9806 on
the full-range task. The symbolic layer generalized essentially perfectly; it was
handed a broken perceptual input.

Report both numbers: 0.5868 is the honest headline, 0.9770 is what actually
measures the symbolic layer's generalization.

## 5d. Capping sums at 9 starves the high digits

Digit frequency in the sums 0–9 training set:

```
1: 18.15%   0: 17.65%   2: 14.15%   3: 12.88%   4: 10.44%
5:  8.23%   6:  7.17%   7:  5.92%   8:  3.67%   9:  1.75%
```

A 9 can only pair with a 0 under that cap, so it appears **ten times less often
than a 1** — while the test set (sums 10–18) consists entirely of large digits.
The experiment imposes a severe distribution shift on top of the compositional
one. This is inherent to the experimental design, not a flaw in the model.

Note this is a **data scarcity** failure, not the hedging failure from finding
#1: the distribution over a 9 is sharp (P(7)=0.942), just wrong. The NLL fix
held. These two failure modes look identical in the accuracy column and
completely different in the softmax — always check the distribution, not just
the argmax.

---

## Architecture and mechanics

## 6. An untrained pipeline outputs a constant ≈ 8.8

Random weights → near-uniform softmax → expected sum ≈ 9.0 (mean of 0–9 is 4.5,
doubled). Confirmed by `symbolic.py`, which returns exactly 9.0000 for uniform
inputs.

**Use this as a diagnostic.** If training plateaus with every prediction stuck
near 9 regardless of input, the CNN is not learning and gradients are not
reaching it.

## 7. One shared CNN, not two

The pair image is split into two 28×28 crops and passed through the **same**
CNN. Two separate networks would give two position-dependent recognizers, each
training on half the digits, neither able to read a digit in the position it
never saw.

Verified by parameter count: the whole solver has **108,618** parameters,
identical to one `ConvNet`. Two CNNs would show 217,236. This is asserted in
`pipeline.py` so a refactor can't silently break it.

## 8. `register_buffer`, not `nn.Parameter`, for the symbolic tables

A `Parameter` would be collected by `model.parameters()` and handed to Adam,
which would happily start **learning the addition table** — at which point the
symbolic module has weights and the project stops being neurosymbolic.

A buffer still moves with `.to(device)` and is saved in the `state_dict`, but is
invisible to the optimizer. `symbolic.py` asserts a parameter count of 0.

## 9. `forward()` returns logits; softmax lives in the pipeline

`CrossEntropyLoss` applies log-softmax internally, so feeding it probabilities
double-applies the transform and silently weakens gradients. Keeping the CNN on
logits lets the same `cnn.py` serve Tier 0 (logits → `CrossEntropyLoss`) and
Tier 1 (logits → softmax → symbolic module) unchanged.

## 10. Adding a buffer breaks strict checkpoint loading

Introducing `sum_onehot` made every earlier checkpoint fail with
`Missing key(s) in state_dict`. Buffers are constants rebuilt by `__init__`, so
they are safe to skip — but weights are not:

```python
missing, _ = solver.load_state_dict(state, strict=False)
assert not [k for k in missing if not k.startswith("symbolic.")], missing
```

Blanket `strict=False` would let a half-loaded CNN pass as a real result.

## 11. Seed the data *and* the model

`MNISTPairs` takes a seed, so pair sampling is reproducible — but
`ConvNet.__init__` is unseeded, so weights differ every run. Both matter for the
Tier 1 comparison: any accuracy gap between the neurosymbolic model and the
baseline regressor must come from architecture, not from one having drawn easier
data or a luckier initialisation. `torch.manual_seed(0)` at the top of `train`
covers weight init and `DataLoader` shuffling.

## 12. The symbolic layer anchors the meaning of each output neuron

Natural question: with no digit labels, how does the CNN know output index 3
means *three* rather than some arbitrary permutation?

`sum_table[i,j] = i+j` asserts that index `i` denotes digit `i`. A CNN that
mapped threes onto index 5 would produce systematically wrong sums and be
penalised. The symbolic module is not only doing arithmetic — it is fixing the
semantics of every output class. No relabelling of 0–9 preserves all sums, so
the mapping is uniquely pinned.

---

## Tier 0 baseline (for reference)

- CNN: 2 conv (1→16→32, 3×3, no padding) + 2×2 pool each + fc 800→128→10
- Shape arithmetic: 28 → 26 → 13 → 11 → **5**, so `fc1.in_features = 32·5·5 = 800`
  (the 11→5 pool floors away a row and column; guessing 6 here is the classic slip)
- 5 epochs, Adam lr=1e-3 → **98.98%** MNIST test accuracy
- Test accuracy dipping slightly in later epochs while train loss keeps falling
  is mild overfitting; at ~5 images out of 10,000 it is noise, not a bug
