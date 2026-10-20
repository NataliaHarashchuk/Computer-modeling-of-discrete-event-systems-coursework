import math


class Element:
    event_label = "подія"

    def __init__(self, name, event_priority=math.inf, rand=None, distribution="constant",
                 delay_mean=0.0, delay_deviation=0.0):
        self.name = name
        self.event_priority = event_priority
        self.rand = rand
        self.distribution = distribution
        self.delay_mean = delay_mean
        self.delay_deviation = delay_deviation
        self.next_time = math.inf
        self.current_time = 0.0
        self.next_element = None

    def get_delay(self):
        return self.delay_with_mean(self.delay_mean)

    def delay_with_mean(self, mean):
        if self.distribution == "exponential":
            return self.rand.exponential(mean)
        if self.distribution == "uniform":
            return self.rand.uniform(mean - self.delay_deviation, mean + self.delay_deviation)
        if self.distribution == "normal":
            return self.rand.normal(mean, self.delay_deviation)
        return self.rand.constant(mean)

    def in_act(self, packet):
        pass

    def out_act(self):
        return None

    def update_statistics(self, delta):
        pass

    def reset_statistics(self):
        pass
