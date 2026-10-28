from simulation.loss_window import LossWindow

COUNT_TOLERANCE = 1e-9


class DestructionLimitMonitor:
    def __init__(self, window_size, destruction_limit):
        self.destruction_limit = destruction_limit
        self.window = LossWindow(window_size)
        self.is_violated = False
        self.evaluated_exits_count = 0
        self.violating_exits_count = 0
        self.violation_time = 0.0
        self.max_destroyed_fraction = 0.0

    @property
    def violating_exits_fraction(self):
        if self.evaluated_exits_count == 0:
            return 0.0
        return self.violating_exits_count / self.evaluated_exits_count

    def exceeds_limit(self):
        return self.window.destroyed_count > self.destruction_limit * self.window.size + COUNT_TOLERANCE

    def on_packet_exit(self, packet):
        self.window.add(packet.destroyed)
        if not self.window.is_full:
            return
        self.is_violated = self.exceeds_limit()
        self.evaluated_exits_count += 1
        if self.is_violated:
            self.violating_exits_count += 1
        self.max_destroyed_fraction = max(self.max_destroyed_fraction, self.window.destroyed_fraction)

    def update_statistics(self, delta):
        if self.is_violated:
            self.violation_time += delta

    def reset_statistics(self):
        self.evaluated_exits_count = 0
        self.violating_exits_count = 0
        self.violation_time = 0.0
        self.max_destroyed_fraction = self.window.destroyed_fraction if self.window.is_full else 0.0
