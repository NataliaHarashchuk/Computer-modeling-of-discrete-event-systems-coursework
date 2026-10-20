from simulation.element import Element


class Sink(Element):
    def __init__(self, name, event_priority, destruction_threshold, time_tolerance):
        super().__init__(name, event_priority)
        self.destruction_threshold = destruction_threshold
        self.time_tolerance = time_tolerance
        self.delivered_count = 0
        self.destroyed_count = 0
        self.accelerated_packets_count = 0
        self.sojourn_time_sum = 0.0
        self.exit_listeners = []

    @property
    def exited_count(self):
        return self.delivered_count + self.destroyed_count

    @property
    def mean_sojourn_time(self):
        if self.exited_count == 0:
            return 0.0
        return self.sojourn_time_sum / self.exited_count

    def add_exit_listener(self, listener):
        self.exit_listeners.append(listener)

    def in_act(self, packet):
        packet.exit_time = self.current_time
        packet.destroyed = packet.sojourn_time > self.destruction_threshold + self.time_tolerance
        self.sojourn_time_sum += packet.sojourn_time
        if packet.was_accelerated:
            self.accelerated_packets_count += 1
        if packet.destroyed:
            self.destroyed_count += 1
        else:
            self.delivered_count += 1
        for listener in self.exit_listeners:
            listener(packet)

    def reset_statistics(self):
        self.delivered_count = 0
        self.destroyed_count = 0
        self.accelerated_packets_count = 0
        self.sojourn_time_sum = 0.0
