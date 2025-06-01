# import numpy as np

# # def count_states(history: np.array(int)):
# #         arr = history.copy()
# #         flat = arr.flatten()
# #         uniques, inverse = np.unique(flat, return_inverse=True)
# #         dense_arr = inverse.reshape(arr.shape)
# #         results = []

# #         for row in dense_arr:
# #             counts = np.bincount(row)
# #             count_dict = {uniques[i] : count for i, count in enumerate(counts) if count > 0}
# #             results.append(count_dict)
# #         return results
