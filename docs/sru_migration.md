# SRU Migration Status

The validated SRU implementation from `/workspace/sru_new/newrl_sru` has been copied into:

- `instinct_nav/instinct_nav/models/sru/`

Current status:

- code is local to `instinct_nav` now
- bootstrap training still uses feed-forward `ActorCritic` for stability
- SRU will be wired back after the true navigation observation/task stack replaces the bootstrap locomotion env

Reason:

- the copied `ActorCriticSRU` targets the newer `rsl_rl` TensorDict/obs_groups API
- the current bootstrap training path in `instinct_nav` is still using `instinct_rl` runner
- forcing them together immediately would mix two migration axes again
