from simulation.element import Element
from simulation.packet import Packet


class Create(Element):
    event_label = "надходження"

    def __init__(self, name, event_priority, interval_generator):
        super().__init__(name, event_priority)
        self.interval_generator = interval_generator
        self.created_count = 0
        self.total_created_count = 0
        self.next_time = self.interval_generator()

    def out_act(self):
        self.total_created_count += 1
        self.created_count += 1
        packet = Packet(id=self.total_created_count, arrival_time=self.current_time)
        self.next_time = self.current_time + self.interval_generator()
        self.next_element.in_act(packet)
        return packet

    def reset_statistics(self):
        self.created_count = 0

    def describe_state(self):
        return f"{self.name}: створено {self.created_count}"
