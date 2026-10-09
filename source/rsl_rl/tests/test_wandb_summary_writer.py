from types import SimpleNamespace

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rsl_rl.utils import wandb_utils


def test_iteration_scalars_are_not_dropped_by_early_commit(tmp_path, monkeypatch):
    rows = {}
    current_step = 0

    def log(data, step=None, commit=True):
        nonlocal current_step
        step = current_step if step is None else step
        assert step >= current_step, "WandB would discard this row"
        current_step = step
        rows.setdefault(step, {}).update(data)
        if commit:
            current_step += 1

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(name=""))
    monkeypatch.setattr(wandb_utils.wandb, "log", log)
    monkeypatch.delenv("WANDB_USERNAME", raising=False)
    writer = wandb_utils.WandbSummaryWriter(str(tmp_path), 1, {"wandb_project": "test"})
    try:
        for step in range(3):
            writer.add_scalar("Loss/future_occupancy", 0.5, step)
            writer.add_scalar("Metrics/robot_goal/success_rate", 0.9, step)
        writer.flush()
        for step in range(3):
            assert rows[step]["Loss/future_occupancy"] == 0.5
            assert rows[step]["Metrics/robot_goal/success_rate"] == 0.9
        events = EventAccumulator(str(tmp_path)).Reload()
        assert [entry.step for entry in events.Scalars("Loss/future_occupancy")] == [0, 1, 2]
    finally:
        writer.close()
