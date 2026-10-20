class EarlyDiscard:
    def __init__(self, destruction_threshold, time_tolerance, discard_element, downstream_channel=None):
        self.destruction_threshold = destruction_threshold
        self.time_tolerance = time_tolerance
        self.discard_element = discard_element
        self.downstream_channel = downstream_channel

    def remaining_downstream_time(self):
        if self.downstream_channel is None:
            return 0.0
        return self.downstream_channel.current_minimum_service_time()

    def is_doomed(self, packet, current_time, service_duration):
        earliest_sojourn_time = (
            current_time - packet.arrival_time + service_duration + self.remaining_downstream_time()
        )
        return earliest_sojourn_time > self.destruction_threshold + self.time_tolerance

    def discard(self, packet):
        packet.discarded = True
        self.discard_element.in_act(packet)
