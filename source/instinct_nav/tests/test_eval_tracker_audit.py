from types import SimpleNamespace

import torch

from scripts.eval_tracker_audit import TrackerAudit


def test_missing_stale_and_duplicate_records_are_reported_without_mutation():
    tracker = SimpleNamespace(write_index=torch.tensor([1, 1, 0]),
                              buffer=torch.tensor([[1., -1., -1.], [1., -1., -1.], [-1., -1., -1.]]),
                              buffer_size=3)
    audit = TrackerAudit(tracker)
    # Env 0 ends without writing; env 1 writes failure then success; env 2 writes failure.
    tracker.write_index[:] = torch.tensor([1, 3, 1])
    tracker.buffer[1, 1:] = torch.tensor([0., 1.])
    tracker.buffer[2, 0] = 0.
    original = tracker.buffer.clone()
    audit.record(torch.arange(3), torch.tensor([False, False, False]))
    result = audit.summary()
    assert result["missing_record"] == 1
    assert result["stale_success"] == 1
    assert result["multiple_records"] == 1
    assert result["disagreements"] == 2
    assert torch.equal(original, tracker.buffer)
    tracker.write_index[0] += 1
    tracker.buffer[0, 1] = 0.
    audit.record(torch.tensor([0]), torch.tensor([False]))
    assert audit.summary()["missing_record"] == 1
