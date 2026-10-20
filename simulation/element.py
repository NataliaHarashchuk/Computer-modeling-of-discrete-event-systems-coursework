import math


class Element:
    event_label = "подія"

    def __init__(self, name, event_priority):
        self.name = name
        self.event_priority = event_priority
        self.next_time = math.inf
        self.current_time = 0.0
        self.next_element = None

    def in_act(self, packet):
        pass

    def out_act(self):
        return None

    def update_statistics(self, delta):
        pass

    def reset_statistics(self):
        pass
