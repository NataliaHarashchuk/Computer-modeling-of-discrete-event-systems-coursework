import math
from collections import deque

from simulation.element import Element


class Process(Element):
    event_label = "кінець обслуговування"

    def __init__(self, name, event_priority, resource_state, service_time_generator):
        super().__init__(name, event_priority)
        self.resource_state = resource_state
        self.service_time_generator = service_time_generator
        self.queue = deque()
        self.current_packet = None
        self.started_services_count = 0
        self.accelerated_services_count = 0
        self.queue_length_integral = 0.0
        self.busy_time = 0.0
        self.max_queue_length = 0
        self.waiting_time_sum = 0.0

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
        if self.started_services_count == 0:
            return 0.0
        return self.waiting_time_sum / self.started_services_count

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
        if self.queue:
            self.start_service(self.queue.popleft())
        return finished_packet

    def start_service(self, packet):
        accelerated = self.resource_state.accelerated
        duration = self.service_time_generator(accelerated)
        record = packet.node_record(self.name)
        record.service_start_time = self.current_time
        self.waiting_time_sum += record.waiting_time
        record.service_duration = duration
        record.accelerated = accelerated
        self.current_packet = packet
        self.next_time = self.current_time + duration
        self.started_services_count += 1
        if accelerated:
            self.accelerated_services_count += 1

    def update_statistics(self, delta):
        self.queue_length_integral += self.queue_length * delta
        if self.is_busy:
            self.busy_time += delta

    def reset_statistics(self):
        self.started_services_count = 0
        self.accelerated_services_count = 0
        self.queue_length_integral = 0.0
        self.busy_time = 0.0
        self.max_queue_length = self.queue_length
        self.waiting_time_sum = 0.0

    def describe_state(self):
        channel = "зайнятий" if self.is_busy else "вільний"
        return f"{self.name}: черга {self.queue_length}, канал {channel}"
