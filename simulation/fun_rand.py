import math

import numpy as np


class FunRand:
    def __init__(self, seed=None):
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    def standard_uniform(self):
        return float(self.generator.random())

    def exponential(self, mean):
        return -mean * math.log(1.0 - self.standard_uniform())

    def uniform(self, minimum, maximum):
        return minimum + (maximum - minimum) * self.standard_uniform()

    def constant(self, value):
        return float(value)
