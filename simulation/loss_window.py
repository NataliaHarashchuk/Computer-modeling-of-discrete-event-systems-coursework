from collections import deque


class LossWindow:
    def __init__(self, size):
        self.size = size
        self.outcomes = deque(maxlen=size)
        self.destroyed_count = 0

    def __len__(self):
        return len(self.outcomes)

    @property
    def is_full(self):
        return len(self.outcomes) == self.size

    @property
    def destroyed_fraction(self):
        if not self.outcomes:
            return 0.0
        return self.destroyed_count / len(self.outcomes)

    def add(self, destroyed):
        outcome = 1 if destroyed else 0
        if self.is_full:
            self.destroyed_count -= self.outcomes[0]
        self.outcomes.append(outcome)
        self.destroyed_count += outcome
