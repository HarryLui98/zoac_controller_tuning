import ray
import numpy as np


@ray.remote
def create_shared_noise(seed=12345):
    np.random.seed(seed)
    count = 250000000
    noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    return noise


class SharedNoiseTable(object):
    def __init__(self, noise, seed=123):
        np.random.seed(seed)
        self.rg = np.random.RandomState(seed)
        self.noise = noise
        assert self.noise.dtype == np.float32

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return self.rg.randint(0, len(self.noise) - dim + 1)

    def get_delta(self, dim):
        idx = self.sample_index(dim)
        return idx, self.get(idx, dim)
