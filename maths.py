from itertools import product
import random
import itertools

def all_groupings_formatted(n):
    def generate_groups(arr):
        if len(arr) <= 1:
            return [[arr]]
        
        groups = []
        for i in range(1, len(arr)):
            first = [arr[:i]]
            rest = generate_groups(arr[i:])
            for r in rest:
                groups.append(first + r)
        return groups

    def convert_format(grouped_list, n):
        index_map = {}
        group_idx = 0
        for group in grouped_list:
            for elem in group:
                index_map[elem] = group_idx
            group_idx += 1
        return [index_map[i] for i in range(n)]
    
    order = list(range(n))
    all_groupings = generate_groups(order)
    return [convert_format(grouping, n) for grouping in all_groupings]

# print(all_groupings_formatted(4))



def sum_random_nums_n(n):
    """Randomly split integer n into a list of positive integers whose sum is n."""
    k = random.randint(1, n)  # Number of parts

    # Random cut points between 1 and n-1
    cuts = sorted(random.sample(range(1, n), k - 1))
    boundaries = [0] + cuts + [n]

    # Compatible with Python < 3.10 (instead of itertools.pairwise)
    parts = [b - a for a, b in zip(boundaries, boundaries[1:])]

    return parts

# for i in range(10):
#     # Example usage:
#     n = 10
#     print(sum_random_nums_n(n))



