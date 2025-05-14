def get_integer_functions(f: list[bool], extra_functions: list[int]) -> int:
    """
    Converts list of bools to 32-bit int with bits representing truth table.

    :param f: List of boolean values representing the truth table.
    :type f: List[bool]
    :param extra_functions: List to store extra functions if the truth table exceeds 32 bits.
    :type extra_functions: List[int]
    :returns: Integer representation of the truth table.
    :rtype: int
    """
    retval = 0
    i = 0
    prefix = 0
    tempLen = len(f)
    if tempLen > 32:
        for i in range(32):
            if f[i + prefix]:
                retval |= 1 << i

        prefix += 32
        tempLen -= 32

    else:
        for i in range(tempLen):
            if f[i]:
                retval |= 1 << i

        return retval

    while tempLen > 0:
        other = 0
        for i in range(32):
            if f[i + prefix]:
                other |= 1 << i

        prefix += 32
        tempLen -= 32
        extra_functions.append(other)

    return retval
