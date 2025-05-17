import numpy as np


def merge_attractors(attractors):
        sets = [set(arr) for arr in attractors]
        merged = []
        while sets:
            current, *rest = sets
            current = set(current)
            changed = True
            while changed:
                changed = False
                remaining = []
                for s in rest:
                    if current & s:  # If there is any overlap
                        current |= s
                        changed = True
                    else:
                        remaining.append(s)
                rest = remaining
            merged.append(np.array(list(current)))
            sets = rest
        return merged