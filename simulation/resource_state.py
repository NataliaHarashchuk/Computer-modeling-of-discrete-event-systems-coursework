class ResourceState:
    def __init__(self, accelerated=False):
        self.accelerated = accelerated
        self.accelerated_time = 0.0

    def update_statistics(self, delta):
        if self.accelerated:
            self.accelerated_time += delta

    def reset_statistics(self):
        self.accelerated_time = 0.0
