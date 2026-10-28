from simulation.loss_window import LossWindow


class ResourceController:
    def __init__(self, resource_state, window_size, activation_threshold, deactivation_threshold):
        self.resource_state = resource_state
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.window = LossWindow(window_size)
        self.activation_count = 0
        self.deactivation_count = 0

    def on_packet_exit(self, packet):
        self.window.add(packet.destroyed)
        if not self.window.is_full:
            return
        fraction = self.window.destroyed_fraction
        if not self.resource_state.accelerated and fraction >= self.activation_threshold:
            self.resource_state.accelerated = True
            self.activation_count += 1
        elif self.resource_state.accelerated and fraction < self.deactivation_threshold:
            self.resource_state.accelerated = False
            self.deactivation_count += 1

    def reset_statistics(self):
        self.activation_count = 0
        self.deactivation_count = 0
