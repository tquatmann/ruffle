# ruffle
Transforms a Markov model (DTMC/MDP) into related models — interval models (IMDPs) learned or
constructed from it, or models perturbed by sampling — for experimenting with model-learning and
robustness questions.

Built on top of the C++ API of [Storm](https://www.stormchecker.org).

## Getting Started
Before starting, make sure that Storm is installed. If not, see the [documentation](https://www.stormchecker.org/documentation/obtain-storm/build.html) for details on how to install Storm. It is necessary to build Storm from source, i.e. a Homebrew installation will most likely not work.

First, configure and compile the project. Therefore, execute
```
mkdir build
cd build
cmake ..
make
cd ..
```

## Usage
```
./build/bin/ruffle --input <file> --output <file> --mode <mode> [options]
```

`--input` and `--output` accept [UMB](https://pmc-tools.github.io/umb/) or [DRN](https://www.stormchecker.org/documentation/background/languages.html#the-drn-format)
model files; the format is detected from each file's extension independently, so input and output can
use different formats:
- `.umb`
- `.drn`, `.drn.gz`, `.drn.xz` (gzip/xz-compressed DRN)

`--mode` selects what should be done to the model:

- **`learn-interval`** — learn an IMDP by treating the input model as a black-box system under
  learning: each state-action pair is repeatedly sampled and the observed successor frequencies are
  turned into Clopper-Pearson confidence intervals on the true transition probabilities. Pick exactly
  one sampling strategy:
  - `--samples <uint>` — draw exactly this many samples per state-action pair.
  - `--delta <double>` — sample each state-action pair until the feasible L1 diameter of its
    successor confidence polytope is at most this value.

  Also supports:
  - `--lambda <double>` — the local successor-distribution failure probability, distributed over
    each state-action pair's successors (default: 0.01).
  - `--full-coverage` — keep sampling a state-action pair past the usual stopping point until every
    one of its successors with real probability > 0 has been sampled at least once. Avoids a rare
    successor that never gets sampled ending up with a lower bound of exactly 0.
  - `--seed <uint64>` sets the random number generator seed. If omitted, a random seed is drawn.

- **`widen-interval`** — deterministically replaces every concrete probability strictly between 0 and
  1 by an interval of width `--delta` (required), centered at that probability and clamped to
  `[0, 1]`. No sampling involved.
  - `--epsilon <double>` — raises every lower interval bound to at least `min(p, epsilon)`, where `p`
    is the real probability of that transition.

- **`sample-distribution`** — samples the real distribution of each state-action pair and replaces it
  by the empirical distribution over the samples (a perturbed point model, not an interval model).
  Pick exactly one sampling strategy:
  - `--samples <uint>` — draw exactly this many samples per state-action pair.
  - `--delta <double>` — sample each state-action pair until the L1 distance between the real and
    the empirical distribution is at most this value.

  Also supports:
  - `--full-coverage` and `--seed <uint64>` — same as for `learn-interval` above.

- **`perturb-distribution`** — adds noise to the distribution of each state-action pair (a perturbed
  point model, not an interval model), following the perturbation of large models in the experiments
  of Kiefer and Tang, *Approximate Bisimulation Minimisation* (FSTTCS 2021): for each state-action
  pair, an error is drawn that is uniformly distributed in `[0, delta]` with probability `1 - lambda`
  and `min(2 * delta, 1)` otherwise. Half of it is added to randomly chosen successors and half of it
  is subtracted from randomly chosen successors, so the L1 distance to the real distribution is at
  most the error. Transitions are neither added nor removed, and state-action pairs with a single
  successor are kept. In contrast to `sample-distribution`, no successors are sampled, so the runtime
  does not depend on `delta`.
  - `--delta <double>` — required; the L1 distance (called epsilon in the paper).
  - `--lambda <double>` — the probability with which the L1 distance is `2 * delta` instead (called
    delta in the paper; default: 0.01).
  - `--seed <uint64>` — same as for `learn-interval` above.

Run `./build/bin/ruffle --help` for the full option list.

### Examples
```
./build/bin/ruffle --input model.umb --output model-learned.umb --mode learn-interval --samples 10000 --seed 42
./build/bin/ruffle --input model.umb --output model-widened.umb --mode widen-interval --delta 0.1
./build/bin/ruffle --input model.umb --output model-perturbed.umb --mode sample-distribution --samples 10000 --seed 42
./build/bin/ruffle --input model.umb --output model-perturbed.umb --mode perturb-distribution --delta 0.0001 --seed 42
./build/bin/ruffle --input model.drn --output model-widened.drn.gz --mode widen-interval --delta 0.1
```
