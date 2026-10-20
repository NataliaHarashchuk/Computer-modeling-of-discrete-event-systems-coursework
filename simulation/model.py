class Model:
    def __init__(self, elements, time_tolerance):
        self.elements = list(elements)
        self.time_tolerance = time_tolerance
        self.statistics_collectors = list(self.elements)
        self.event_listeners = []
        self.current_time = 0.0
        self.statistics_start_time = 0.0
        self.event_count = 0

    def add_statistics_collector(self, collector):
        self.statistics_collectors.append(collector)

    def add_event_listener(self, listener):
        self.event_listeners.append(listener)

    @property
    def next_event_time(self):
        return min(element.next_time for element in self.elements)

    @property
    def observed_time(self):
        return self.current_time - self.statistics_start_time

    def advance_time_to(self, new_time):
        new_time = max(new_time, self.current_time)
        delta = new_time - self.current_time
        for collector in self.statistics_collectors:
            collector.update_statistics(delta)
        self.current_time = new_time
        for element in self.elements:
            element.current_time = new_time

    def reset_statistics(self):
        for collector in self.statistics_collectors:
            collector.reset_statistics()
        self.statistics_start_time = self.current_time

    def find_due_elements(self):
        return [
            element for element in self.elements
            if abs(element.next_time - self.current_time) <= self.time_tolerance
        ]

    def process_simultaneous_events(self):
        while True:
            due_elements = self.find_due_elements()
            if not due_elements:
                return
            element = min(due_elements, key=lambda candidate: candidate.event_priority)
            packet = element.out_act()
            self.event_count += 1
            for listener in self.event_listeners:
                listener(element, packet)

    def run_until(self, stop_time, include_events_at_stop_time=True):
        while True:
            next_event_time = self.next_event_time
            if next_event_time > stop_time or (next_event_time == stop_time and not include_events_at_stop_time):
                break
            self.advance_time_to(next_event_time)
            self.process_simultaneous_events()
        self.advance_time_to(stop_time)

    def simulate(self, simulation_time):
        self.run_until(simulation_time)
