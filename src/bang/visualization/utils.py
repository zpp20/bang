import numpy as np

from bang.core.pbn.utils.state_printing import convert_to_binary_representation


def max_bit_length(
    trajectory: np.ndarray,
) -> int:
    """
    Calculate the maximum bit length required to represent the trajectory.
    :param trajectory: The trajectory to analyze, where each row represents a state.
    :type trajectory: np.ndarray
    :return: The maximum bit length required to represent the trajectory.
    :rtype: int
    """

    # Find the maximum value in the trajectory
    max_value = np.max(trajectory)

    # Calculate the number of bits needed to represent the maximum value
    # If max_value is 0, we need 1 bit, otherwise calculate the bit length
    if max_value == 0:
        return 1
    else:
        return int(np.floor(np.log2(max_value))) + 1


def convert_to_ones_zeros(
    trajectory: np.ndarray,
    bit_length: int,
) -> str:
    bool_repr = convert_to_binary_representation(trajectory, bit_length)

    ones_zeros = np.array(bool_repr, dtype=np.uint32)[0]

    string_repr = "".join(["1" if bit else "0" for bit in ones_zeros])

    return string_repr
