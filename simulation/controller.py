from collections import deque


class ResourceController:
    def __init__(self, resource_state, window_size, activation_threshold, deactivation_threshold):
        self.resource_state = resource_state
        self.window_size = window_size
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.window = deque(maxlen=window_size)
        self.destroyed_in_window = 0
        self.activation_count = 0
        self.deactivation_count = 0

    @property
    def destroyed_fraction(self):
        if not self.window:
            return 0.0
        return self.destroyed_in_window / len(self.window)

    def on_packet_exit(self, packet):
        if len(self.window) == self.window_size:
            self.destroyed_in_window -= self.window[0]
        self.window.append(1 if packet.destroyed else 0)
        self.destroyed_in_window += 1 if packet.destroyed else 0
        fraction = self.destroyed_fraction
        if not self.resource_state.accelerated and fraction >= self.activation_threshold:
            self.resource_state.accelerated = True
            self.activation_count += 1
        elif self.resource_state.accelerated and fraction < self.deactivation_threshold:
            self.resource_state.accelerated = False
            self.deactivation_count += 1

    def reset_statistics(self):
        self.activation_count = 0
        self.deactivation_count = 0
