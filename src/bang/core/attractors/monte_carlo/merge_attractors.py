from collections import defaultdict

import numpy as np


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def merge_attractors(attractors):
    attractor_sets = [set(attractor) for attractor in attractors]
    uf = UnionFind()

    for attractor in attractor_sets:
        it = iter(attractor)
        first = next(it)

        for item in it:
            uf.union(first, item)

    merged_attractors = defaultdict(set)
    for attractor in attractor_sets:
        for state in attractor:
            root = uf.find(state)
            merged_attractors[root].add(state)

    return [np.array(list(merged_attractor)) for merged_attractor in merged_attractors.values()]
