from simulation.element import Element
from simulation.packet import Packet


class Create(Element):
    event_label = "надходження"

    def __init__(self, name, event_priority, rand, distribution, delay_mean, delay_deviation=0.0):
        super().__init__(name, event_priority, rand, distribution, delay_mean, delay_deviation)
        self.created_count = 0
        self.total_created_count = 0
        self.next_time = self.get_delay()

    def out_act(self):
        self.total_created_count += 1
        self.created_count += 1
        packet = Packet(id=self.total_created_count, arrival_time=self.current_time)
        self.next_time = self.current_time + self.get_delay()
        self.next_element.in_act(packet)
        return packet

    def reset_statistics(self):
        self.created_count = 0
