# Method

Everything here is a toy abstraction built for a CPU. It is not a link-level or system-level simulator, and none of
its numbers can be compared with field-trial results.

## Radio model ([`aran/phy.py`](../aran/phy.py), [`aran/env.py`](../aran/env.py))

| Piece | Model | Notes |
|---|---|---|
| Layout | 3 controlled cells on a triangle (ISD 500 m) + 6 uncontrolled outer cells on a ring at 1000 m | `coupled` scenario: no outer ring |
| UEs | 4 per cell, uniform in a disc of radius 0.6 ISD (0.9 in `coupled`); redrawn until the serving cell is the strongest site | attachment by long-term gain, no handover |
| Path loss | 128.1 + 37.6 log10(d/km), d >= 35 m (TR 36.814 macro) | |
| Shadowing | 8 dB log-normal, independent per (UE, site) link | no inter-site correlation |
| Fast fading | flat Rayleigh, Gauss-Markov with lag-1 correlation J0(2 pi f_d T), 3.5 GHz, 1 ms TTI | no frequency selectivity, one PRB group |
| Outer cells | two-state on/off Markov, P(on) = 0.5, mean burst 20 TTI, full power when on | "flash-light" interference |
| Noise | -174 dBm/Hz + 10 log10(10 MHz) + 9 dB NF = -95 dBm | |
| MCS | 15 entries = LTE 4-bit CQI table efficiencies (0.15 ... 5.55 bit/s/Hz) | |
| BLER | logistic in SINR, 10% at the usual CQI switching threshold, slope 2 /dB | no HARQ combining / retransmissions |
| CQI | measured every TTI on full-power reference signals with the neighbours' actual interference, + N(0, 1 dB) noise, quantised, delivered `cqi_delay` = 4 TTIs late | wideband, periodic |
| Scheduler | proportional fair (EWMA window 50 TTI) on the reported CQI, one UE per cell per TTI, full buffer | |

## Action, observation, reward

- **Action** (per cell, 21 discrete): MCS delta in {-4..+2} relative to the reported CQI index, times power back-off
  in {0, -3, -6} dB.
- **Observation** (12 floats per cell): reported CQI, a 10%-target OLLA offset the env tracks for the UE, the UE's
  wideband CQI average, CQI trend, last 4 HARQ outcomes, own last power, neighbours' mean last power, neighbours'
  scheduled-UE wideband CQI (an idealised X2-style exchange), top-CQI flag. All learners get the same observation.
- **Reward**: the cell's goodput EFF[MCS] x ACK (bit/s/Hz). IQL uses its own cell's goodput, IQL-team the mean over
  cells, VDN the sum (in one joint TD target). The single-agent DQN uses its own goodput.

## Methods

| Method | What it is | Tuned on |
|---|---|---|
| Fixed MCS | one MCS for everyone | dev episodes (best of 15) |
| CQI only | inner loop: MCS = reported CQI | - |
| OLLA 10% | outer loop, BLER target 10%, 0.5 dB down-step, offset starts at 0 | textbook default |
| OLLA tuned | outer loop, (target, step, initial offset) picked on the dev episodes from 5 x 4 x 6 = 120 settings | dev episodes |
| OLLA tuned + back-off | tuned OLLA; a cell uses -6 dB when its UE reports the top CQI and OLLA picks the top MCS | dev (inherits) |
| Genie | knows the true current SINR, all cells at full power; picks the goodput-maximising MCS | non-causal reference, not an upper bound once power is controllable |
| Single-agent DQN | learns in cell 0 while cells 1-2 run tuned OLLA at full power, then deployed in every cell | - |
| IQL | parameter-shared independent DQN, own-cell reward | - |
| IQL team | parameter-shared independent DQN, team (mean) reward | - |
| VDN | parameter-shared DQN, Q_tot = sum of per-cell Q, summed team reward (Sunehag et al., 2018) | - |

The learners share one set of hyperparameters (2 x 64 ReLU MLP, ~6k parameters, Adam 5e-4, double DQN, Huber
loss, gamma 0.5, batch 128, replay 100k, target sync every 1000 updates, epsilon 1 -> 0.05 over the first 60% of
1000 training episodes of 200 TTIs). They were **not** tuned per method: one pilot run (seed 0, 400 episodes,
scored on dev episodes) only checked that training works. That is a limitation: a per-method search could change
the ranking among learners.

## Data splits

| Split | Episode seeds | Used for |
|---|---|---|
| training | 1,000,000 x (seed + 1) + episode | learners only |
| dev | 5000-5049 | rule-baseline tuning, the one pilot |
| evaluation | 10000-10099 | every reported number |

All randomness of an episode is drawn from its seed up front, so every method sees exactly the same drops, fading,
outer-cell activity, CQI noise and HARQ coin flips (common random numbers). Only interference between the controlled
cells and the PF state depend on actions. `tests/test_agents.py::test_seed_streams_disjoint` checks the split.

## Statistics ([`aran/stats.py`](../aran/stats.py), [`scripts/aggregate.py`](../scripts/aggregate.py))

- Learned methods give a crossed matrix: 15 training seeds x 100 evaluation episodes. Point estimates are the grand
  mean. **95% CIs** use a two-way (pigeonhole) bootstrap that resamples seeds and episodes independently (4000
  replicates). That reflects both training-seed variability and the finite evaluation set. It is slightly
  conservative for crossed designs (it double-counts the interaction term once), so the intervals lean wide.
- **Tests**: for anything involving a learner, the training seed is the unit. We use an exact two-sided sign-flip
  test (all 2^15 sign patterns) on the 15 per-seed mean differences over the shared episodes. Learner-vs-learner
  pairs are matched by seed: they share training-episode seeds and evaluation episodes. Rule-vs-rule comparisons
  are deterministic per episode, so the unit is the episode (Monte-Carlo sign-flip, 100k flips).
- **Multiple comparisons**: 17 pre-registered comparisons per scenario (listed in `COMPARISONS`) x 2 scenarios = one
  Holm family of 34. A comparison is called significant only if Holm p < 0.05 **and** its 95% CI excludes 0.
- The speed x delay sweep is exploratory. It only evaluates, on 30 episodes, and its markers are uncorrected.

## Reproducibility guards

- `scripts/aggregate.py` refuses (exit 2, writes nothing) unless `results/raw` holds exactly the expected files.
  `--allow-partial` is for debugging, and it must write to a separate `--out`.
- `scripts/make_figures.py` and `scripts/readme_tables.py` refuse a summary that does not cover the full grid.
- `scripts/mutation_check.py` applies single-line bugs to a throw-away copy and requires `pytest` to fail on each.
- `figures/` PNGs are deterministic given the CSVs (fixed hash salt, no timestamp metadata).
