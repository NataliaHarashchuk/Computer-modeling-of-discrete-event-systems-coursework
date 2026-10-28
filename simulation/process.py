import math
from collections import deque

from simulation.element import Element


class Process(Element):
    event_label = "кінець обслуговування"

    def __init__(self, name, event_priority, rand, distribution, delay_mean, accelerated_delay_mean, resource_state,
                 early_discard=None):
        super().__init__(name, event_priority, rand, distribution, delay_mean)
        self.accelerated_delay_mean = accelerated_delay_mean
        self.resource_state = resource_state
        self.early_discard = early_discard
        self.queue = deque()
        self.current_packet = None
        self.started_services_count = 0
        self.discarded_count = 0
        self.queue_length_integral = 0.0
        self.busy_time = 0.0
        self.max_queue_length = 0
        self.waiting_time_sum = 0.0
        self.queue_exits_count = 0

    @property
    def queue_length(self):
        return len(self.queue)

    @property
    def is_busy(self):
        return self.current_packet is not None

    def mean_queue_length(self, observation_time):
        if observation_time <= 0:
            return 0.0
        return self.queue_length_integral / observation_time

    def utilization(self, observation_time):
        if observation_time <= 0:
            return 0.0
        return self.busy_time / observation_time

    @property
    def mean_waiting_time(self):
        if self.queue_exits_count == 0:
            return 0.0
        return self.waiting_time_sum / self.queue_exits_count

    def current_minimum_service_time(self):
        if self.distribution != "constant":
            return 0.0
        return self.accelerated_delay_mean if self.resource_state.accelerated else self.delay_mean

    def get_delay(self):
        mean = self.accelerated_delay_mean if self.resource_state.accelerated else self.delay_mean
        return self.delay_with_mean(mean)

    def in_act(self, packet):
        packet.open_node_record(self.name, self.current_time)
        if self.is_busy:
            self.queue.append(packet)
            self.max_queue_length = max(self.max_queue_length, self.queue_length)
        else:
            self.start_service(packet)

    def out_act(self):
        finished_packet = self.current_packet
        self.current_packet = None
        self.next_time = math.inf
        self.next_element.in_act(finished_packet)
        while self.queue and not self.is_busy:
            self.start_service(self.queue.popleft())
        return finished_packet

    def start_service(self, packet):
        accelerated = self.resource_state.accelerated
        duration = self.get_delay()
        record = packet.node_record(self.name)
        self.leave_queue(record)
        if self.early_discard is not None and self.early_discard.is_doomed(packet, self.current_time, duration):
            self.discarded_count += 1
            self.early_discard.discard(packet)
            return
        record.service_start_time = self.current_time
        record.accelerated = accelerated
        self.current_packet = packet
        self.next_time = self.current_time + duration
        self.started_services_count += 1

    def leave_queue(self, record):
        record.queue_exit_time = self.current_time
        self.waiting_time_sum += record.waiting_time
        self.queue_exits_count += 1

    def update_statistics(self, delta):
        self.queue_length_integral += self.queue_length * delta
        if self.is_busy:
            self.busy_time += delta

    def reset_statistics(self):
        self.started_services_count = 0
        self.discarded_count = 0
        self.queue_length_integral = 0.0
        self.busy_time = 0.0
        self.max_queue_length = self.queue_length
        self.waiting_time_sum = 0.0
        self.queue_exits_count = 0
