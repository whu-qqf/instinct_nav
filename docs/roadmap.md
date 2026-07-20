# Roadmap

## Current State

- standalone `instinct_nav` training entrypoint works
- clean-room navigation task scaffold exists
- minimal goal-conditioned single-policy baseline works
- SRU code from `sru_new` has been copied locally
- parkour low-level policy bridge scaffold now exists under `instinct_nav/low_level/`
- low-level default target is the frozen checkpoint `/workspace/hiking_artifacts/g1_parkour_20260709_103722/model_30000.pt`

## Next Steps

1. Keep the current single-policy baseline for debugging and learning.
2. Finish the parkour low-level bridge integration path.
3. Build a hierarchical navigation action term where the high-level policy outputs `base_velocity`.
4. Feed that command plus low-level history into the frozen parkour locomotion policy.
5. After the bridge is stable, migrate SRU onto the high-level policy only.
