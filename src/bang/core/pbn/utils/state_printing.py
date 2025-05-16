import numpy as np


def to_binary_array(value, bit_length=8):
    """Convert an integer to a binary array of a given bit length."""
    return [bool((value >> i) & 1) for i in range(bit_length)]


def convert_to_binary_representation(data, bit_length) -> list:
    """
    Recursively convert all numbers in a nested array (or numpy array)
    into arrays of their binary representation.
    """
    if isinstance(data, (int, float, np.integer, np.floating)):
        if isinstance(data, (float, np.floating)) and not data.is_integer():
            raise ValueError("Cannot convert non-integer float to binary representation.")

        if data < 0:
            raise ValueError("Negative values are not supported for binary representation.")

        return to_binary_array(int(data), bit_length)
    elif isinstance(data, (list, tuple)):
        return [convert_to_binary_representation(item, bit_length) for item in data]
    elif isinstance(data, np.ndarray):
        return [convert_to_binary_representation(item, bit_length) for item in data]
    else:
        raise ValueError("Unsupported data type: {}".format(type(data)))


def convert_from_binary_representation(data) -> list | int:
    """
    Recursively convert binary arrays back into their integer representation.
    """
    if isinstance(data, list) and all(isinstance(bit, bool) for bit in data):
        return sum((1 << i) for i, bit in enumerate(data) if bit)
    elif isinstance(data, (list, tuple)):
        return [convert_from_binary_representation(item) for item in data]
    elif isinstance(data, np.ndarray):
        return [convert_from_binary_representation(item) for item in data]
    else:
        raise ValueError("Unsupported data type: {}".format(type(data)))
