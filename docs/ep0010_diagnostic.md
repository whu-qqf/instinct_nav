# ep_0010 live diagnostic

The September 14 static evaluation's ep_0010 is env 870, terrain level 5,
type 25, terminated by large_pitch_angle. Its GIF sidecar does not preserve
the full initial joint state, initial orientation, actions, or random state.
The diagnostic is a same-seed fresh rollout, not guaranteed exact replay.

`eval.py --live_diagnostic_dir PATH --live_diagnostic_env_id 870
--live_diagnostic_steps 60` enables opt-in capture. The mode ends without a
benchmark summary after the frame budget. Ordinary eval defaults are unchanged.

Files per control step:
- Live RGB PNG, captured without rewriting any robot pose.
- NPZ with raw camera depth, critic ray hits, joint/body state, and contact
  forces including force history.
- JSONL with root pose/vertical velocity, action, episode step, obstacle
  activation, and nine near-root PhysX terrain-only queries at the same XY
  coordinates as the critic height scanner. A missing PhysX hit is explicit.

Ray-query filtering has CPU tests. This does not yet draw a collision-wireframe
overlay or sample every physics step.
Control-rate sampling may miss a brief impact, so contact history is included.

## September 14 execution results

Run root: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-09-11_15-37-00_mask_cnn_encounter_restored_rewards_4096_20260911_153005`.

- Formal evaluation pair `eval_pair_20260914_160402` completed, 8192 episodes
  per condition with 4096 environments. Static legacy success is 85.4736%;
  dynamic difficulty-zero legacy success is 84.0942%.
- The independent episode-outcome audit reports 81.5186% and 79.6997%.
  Static has 878 missing tracker writes and 323 stale successes; dynamic has
  1000 missing writes and 360 stale successes. Keep the legacy score for
  historical comparisons, but do not treat the discrepancy as negligible.
- Initial diagnostic `diagnostic_ep10_20260914_164455` failed on mixed CPU/CUDA
  camera rotation inputs. Passing `device=env.device` fixes that call. The
  failed worker was stopped after formal evaluation had completed.
- `diagnostic_ep10_retry_20260914` finished with exit 0 and saved 60 RGB/NPZ
  frames and state records. It uses 32 environments, remaps the reference
  spawn/goal to env 0, and approximates heading and initial joint state.
  All recorded steps have obstacles disabled and zero activated obstacles.
  Root Z ranges from 0.462 to 0.950 m; vertical velocity from -2.347 to
  +0.379 m/s. The original upward launch was not reproduced.
- PhysX callback errors in that retry invalidate ALL its terrain-query
  results, despite exit 0. The installed API passes `RaycastHit` objects with
  `collision`, `rigid_body`, and `position` attributes, not dictionaries.
  The corrected helper also surfaces callback errors instead of silently
  treating them as missing terrain. Three CPU tests cover terrain filtering,
  missing hits, and callback failure propagation.
- `diagnostic_physx_verified_20260914` saved five frames with the corrected
  callback: all 45 rays hit terrain, with maximum absolute scan/PhysX height
  difference 2.37e-6 m. This validates the query at those sampled locations,
  NOT the original failure area. The first frame's scan is stale, approximately
  747.3 m away from the relocated root. By frame 1 the episode has reset to a
  different location (root XY 135.594, 9.171). Frames 1-4 have near-root scan
  samples and maximum height difference 1.91e-6 m.
- The recorder now includes `nearest_scan_xy_distance_m` so a stale scan is
  explicit. The approximate manifest path needs sensor/cache and terrain
  assignment synchronization before it can support a failure diagnosis.

Reproduction limitations: original full initial state is unavailable; the
manifest application does not restore terrain level/type assignments or all
sensor/memory caches. The geometry digest includes per-env terrain assignments,
so a 32-env versus 4096-env digest difference cannot establish different meshes.
This diagnostic is not an exact replay or a policy benchmark.

## Next gates

1. Obtain valid PhysX/height-scan comparisons and live failure recordings before
   attributing ep_0010 to hidden obstacles, geometry mismatch, or the network.
2. Add exactly-once episode-result accounting as an isolated, tested change.
   Audit overlapping terminations and orientation-only failures; retain the
   historical evaluation metric alongside the corrected metric.
3. Log the instantaneous tracker rate actually used for difficulty promotion,
   its threshold, and promotion decisions. A smoothed training success scalar
   above 0.9 alone does not establish a curriculum bug.
4. Run mask-zero inference ablation with matching checkpoint and evaluation
   protocol after the preceding checks. This measures input sensitivity, not
   the training-time causal benefit of the architecture.
5. Choose a new training experiment only after these checks; preserve the
   obstacle schedule and verified checkpoints for controlled comparisons.
