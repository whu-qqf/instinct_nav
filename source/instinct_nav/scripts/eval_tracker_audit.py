"""Read-only audit of legacy success records at episode boundaries."""

import torch


class TrackerAudit:
    def __init__(self, tracker):
        self.tracker = tracker
        self.start_index = tracker.write_index.clone()
        self.counts = dict(episodes=0, missing_record=0, multiple_records=0,
                           stale_success=0, legacy_successes=0, outcome_successes=0,
                           disagreements=0, counter_reset=0)

    def record(self, done_ids, outcome_success):
        index = self.tracker.write_index[done_ids]
        delta = index - self.start_index[done_ids]
        latest = self.tracker.buffer[done_ids, (index - 1) % self.tracker.buffer_size] > 0
        outcome = outcome_success.to(latest.device)
        values = dict(episodes=len(done_ids), missing_record=(delta == 0).sum().item(),
                      multiple_records=(delta > 1).sum().item(),
                      stale_success=((delta == 0) & latest).sum().item(),
                      legacy_successes=latest.sum().item(), outcome_successes=outcome.sum().item(),
                      disagreements=(latest != outcome).sum().item(),
                      counter_reset=(delta < 0).sum().item())
        for key, value in values.items():
            self.counts[key] += int(value)
        self.start_index[done_ids] = index

    def summary(self):
        n = max(self.counts["episodes"], 1)
        return {**self.counts,
                "legacy_success_rate": self.counts["legacy_successes"] / n,
                "outcome_success_rate": self.counts["outcome_successes"] / n}
