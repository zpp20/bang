import typing

if typing.TYPE_CHECKING:
    from bang.core import PBN


def select_nodes(pbn: "PBN", nodes: list[int]):
    new_F = list()
    new_varF = list()
    new_nv = list()
    for F_func, F_vars, n_vars in zip(pbn._f, pbn._var_f_int, pbn._nv):
        # assumes F contains truthtables for sorted vars
        # print("F_vars ", F_vars)
        new_nv.append(0)
        new_varF.append(list())
        new_F.append(list())
        curr_num_vars = len(F_vars)
        curr_F = F_func
        current_removed = 0
        for i, var in enumerate(F_vars):
            if var not in nodes:
                curr_i = i - current_removed
                var_state = 0
                # indeces = [j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state for j in range(2**(curr_num_vars - 1))]
                # print("indeces - ", indeces, " var_state ", var_state, " curr_i ", curr_i)
                curr_F = [
                    curr_F[j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state]
                    for j in range(2 ** (curr_num_vars - 1))
                ]
                curr_num_vars -= 1
                current_removed += 1
            else:
                new_varF[-1].append(var)
                new_nv[-1] += 1

        new_F[-1].append(curr_F)

    # Translation of old nodes into new nodes
    translation = {node: idx for idx, node in enumerate(nodes)}

    def translate(to_translate: list[int]):
        return [translation[elem] for elem in to_translate]

    # Assumes nodes are numbered from 0 to n-1!!!!!
    n = len(nodes)
    nf = [pbn._nf[idx] for idx, i in enumerate(pbn._nf) if idx in nodes]
    nv = [new_nv[idx] for idx, i in enumerate(new_nv) if idx in nodes]
    F = [sublist[0] for sublist in new_F]
    F = [F[idx] for idx, i in enumerate(F) if idx in nodes]
    print()
    varFInt = [translate(new_varF[idx]) for idx, i in enumerate(new_varF) if idx in nodes]
    cij = pbn._cij
    perturbation = pbn._perturbation
    npNode = [np for np in pbn._np_node if np in nodes]
    npNode.append(n)

    return pbn.clone_with(
        n,
        nf,
        nv,
        F,
        varFInt,
        cij,
        perturbation,
        npNode,
        pbn._n_parallel,
        update_type="synchronous",
    )
