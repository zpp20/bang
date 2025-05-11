import typing

if typing.TYPE_CHECKING:
    from bang.core import PBN


def reduce_F(self: "PBN", states: list[list[int]]) -> tuple:
    # """
    # Reduces truth tables of PBN by removing states that do not change.

    # :param states: List of investigated states. States are lists of int with length n where i-th index represents i-th variable. 0 represents False and 1 represents True.
    # :type states: List[List[int]]
    # :returns: Tuple containing list of indices of variables that change between states and truth tables with removed constant variables.
    # :rtype: tuple
    # """
    initial_state = states[0]

    constant_vars = {i for i in range(0, self._n)}

    for state in states[1:]:
        for var in range(0, self._n):
            if initial_state[var] != state[var]:
                constant_vars.remove(var)

    new_F = list()
    new_varF = list()

    for F_func, F_vars in zip(self._f, self._var_f_int):
        new_varF.append(list())
        new_F.append(list())
        curr_num_vars = len(F_vars)
        curr_F = F_func

        # curr_vars = F_vars

        current_removed = 0
        for i, var in enumerate(F_vars):
            if var in constant_vars:
                curr_i = i - current_removed
                var_state = initial_state[var]
                curr_F = [
                    curr_F[j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state]
                    for j in range(2 ** (curr_num_vars - 1))
                ]
                curr_num_vars -= 1
                current_removed += 1
            else:
                new_varF[-1].append(var)

        new_F[-1].append(curr_F)

    return new_varF, new_F
