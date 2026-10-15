import numpy as np


class FunRand:
    def __init__(self, seed=None):
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    def exponential(self, mean):
        return float(self.generator.exponential(mean))

    def uniform(self, minimum, maximum):
        return float(self.generator.uniform(minimum, maximum))

    def normal(self, mean, deviation):
        return float(self.generator.normal(mean, deviation))

    def erlang(self, mean, k):
        return float(self.generator.gamma(k, mean / k))

    def constant(self, value):
        return float(value)
