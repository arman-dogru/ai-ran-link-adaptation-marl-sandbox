# Learner pilot (the only check made before the main grid)

These are the original files, copied verbatim from the box's `/tmp` (written 08:16-08:18 EDT on 27 Oct 2026,
before `results/dev_tuning_*.json` existed):

- `pilot.py`: the command was `PYTHONPATH=. python pilot.py <method> 400` for `sa`, `iql`, `iql_team` and `vdn`.
- `pilot_<method>.log`: the stdout of each run.

What it did: it trained seed 0 for 400 episodes with the hyperparameters already written in `aran/agents.py`
(lr 5e-4, gamma 0.5, batch 128, eps_frac 0.6, ...). The reference policy was the textbook `OLLA(0.1, 0.5)`, and
the score came from **dev** episodes 5000-5029. Evaluation episodes were not touched.

What it was used for: only to check that the learners learn something (dev SE 1.57-1.75 against about 1.62 for
the 10% OLLA). **No hyperparameter was changed after it.** The values in `agents.py` were set a priori and have
been in that file unchanged since its first commit (a4263aa). Note that the pilot ran against an earlier OLLA
signature (no `init` argument) and before dev tuning, so its numbers are not comparable with `results/`.
