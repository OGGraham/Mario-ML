import numpy as np
import random
from SumTree import SumTree

class Memory:  # stored as ( s, a, r, s, d ) in SumTree
    """
    This class is a modified version of
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    e = 0.01  # prevents error = 0 i.e. no memory has 0 prob of being selected
    a = 0.6  # Amount of randomness in memeory selection -> a = 0 random, a = 1 highest priorities
    beta = 0.4  # Corrects bias of sampling high priority samples over low priority ones (prevents overfitting)
    # (memories sampled more often have decreased priority) -> controls how much these weights affect learning
    beta_increment_per_sampling = 0.001  # Value beta incremented by after each sample (prevents overfitting)

    # Notes for self:
    # Key is error, val is memory (s,a,r,s',d)
    # priority = |error| + e
    # P(i) = p_i^a/sum_k(p_k^a) -> Prob = priority ^ a / sum priorities ^ a
    # Correct high-priority bias w:
    # (1/N * 1/P(i))^b -> b increases over time therefore val decreases

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    # Get priority of sample
    def _get_priority(self, error):
        return (error + self.e) ** self.a

    # Add new entry to tree
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    # Get sample
    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    # Update weights
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
