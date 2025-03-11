import itertools

FORBIDDEN_CHARS = [" ", "\t", "\n", "\r", "\v", "\f", "&", "*", "!", "^", "/", "|", ":", "(", ")"]
# forbidden variable names
FORBIDDEN_NAMES = ["or", "not", "and"]
# for python eval function
LOGICAL_REPLACEMENTS = {"|": " or ", "&": " and ", "!": " not "}


def delete_comments_and_empty_lines(lines):
    new_lines = []
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("//"):
            continue
        new_lines.append(line)
    return new_lines


# extract variable names in the 'fun' function expression  in ASSA format
def get_vars_from_assa_expr(fun, names_dict):
    vars = []
    cur = ""
    started = False

    for i in range(len(fun)):
        # if the character is a forbidden, the variable name is finished
        if fun[i] in FORBIDDEN_CHARS:
            if started:
                if cur not in vars:
                    if cur not in names_dict:
                        raise ValueError("Invalid file format")
                    vars.append(cur)
                cur = ""
                started = False
            else:
                continue
        else:
            cur += fun[i]
            started = True

    if started:
        if cur not in vars:
            if cur in FORBIDDEN_NAMES or cur not in names_dict:
                raise ValueError("Invalid file format")
            vars.append(cur)

    return vars


def get_n(line: str) -> int:
    line = line.strip()

    if not line.startswith("n="):
        raise ValueError("Invalid file format")

    n = line.split("=")[1]

    if not n.isnumeric():
        raise ValueError("Invalid file format")

    n = int(n)

    return n


def get_type(line: str):
    line = line.strip()
    if not line.startswith("type="):
        raise ValueError("Invalid file format")

    type = line.split("=")[1]

    if type not in ["synchronous", "rog", "rmg", "rmgrm", "rmgro", "rmgrorm", "aro"]:
        raise ValueError("Invalid file format")

    return type


def get_perturbation(line):
    line = line.strip()
    if not line.startswith("perturbation="):
        raise ValueError("Invalid file format")
    perturbation = line.split("=")[1]
    try:
        perturbation = float(perturbation)
    except ValueError:
        raise ValueError("Invalid file format")

    return perturbation


def get_variable_names(lines, i, n, names_dict, index_dict):
    for j in range(n):
        name = lines[i].strip()
        if name in names_dict:
            raise ValueError("Duplicate node name")
        if name in FORBIDDEN_NAMES:
            raise ValueError("Invalid node name")
        for char in FORBIDDEN_CHARS:
            if char in name:
                raise ValueError("Invalid node name")

        names_dict[name] = j
        index_dict[j] = name
        i += 1


def checkline(line, expected):
    if line.strip() != expected:
        raise ValueError("Invalid file format")


def get_function(n, lines, probs, names_dict, nv, F, varFInt, i, index_dict):
    fun_expr = [x.strip() for x in lines[i].strip().split(":")]

    # get probability of the function
    try:
        fun_prob = float(fun_expr[0])
        probs.append(fun_prob)
    except ValueError:
        raise ValueError("Invalid file format")

    # extract variable names
    fun = fun_expr[1]
    vars = get_vars_from_assa_expr(fun, names_dict)

    # check if extracted variables are valid
    unsorted_var_indices = []

    for var in vars:
        unsorted_var_indices.append(names_dict[var])

    updated_fun = fun

    for rep in LOGICAL_REPLACEMENTS:
        updated_fun = updated_fun.replace(rep, LOGICAL_REPLACEMENTS[rep])

    sorted_var_indices = sorted(unsorted_var_indices)
    nv.append(len(sorted_var_indices))

    # add variable indices to the list varFInt
    varFInt.append(sorted_var_indices)
    sorted_vars = [index_dict[var] for var in sorted_var_indices]

    # generate every possible variable evaluation
    truth_combinations = list(itertools.product([False, True], repeat=n))
    truth_table = []

    # evaluate the function for every possible combination of variables
    # and store the result in the truth table
    for combination in truth_combinations:
        values = dict(zip(sorted_vars, combination))
        evaluated = eval(updated_fun, {}, values)
        truth_table.append(evaluated)

    F.append(truth_table)


def load_assa(path: str) -> tuple:
    n = 0
    nf = []
    nv = []
    F = []
    varFInt = []
    cij = []
    perturbation = 0.0
    np = []
    # type = ''
    i = 0

    # forbidden characters in the variable names

    names_dict = {}
    index_dict = {}

    with open(path, "r") as f:
        # clean the file
        lines = delete_comments_and_empty_lines(f.readlines())

        # Read the type of the PBN
        get_type(lines[i])
        i += 1

        # Read the number of nodes
        n = get_n(lines[i])
        i += 1

        # Read the perturbation rate
        perturbation = get_perturbation(lines[i])
        i += 1

        # Read the names of the nodes
        checkline(lines[i], "nodeNames")
        i += 1

        get_variable_names(lines, i, n, names_dict, index_dict)
        i += n

        checkline(lines[i], "endNodeNames")
        i += 1

        # for every variable...
        for j in range(n):
            function_count = 0
            checkline(lines[i], "node " + index_dict[j])
            i += 1

            probs = []

            # ... extract the boolean functions for the variable
            while lines[i].strip() != "endNode":
                function_count += 1
                get_function(n, lines, probs, names_dict, nv, F, varFInt, i, index_dict)
                i += 1

            nf.append(function_count)
            cij.append(probs)
            i += 1

        assert len(F) == sum(nf)

        checkline(lines[i], "npNode")
        i += 1

        # get_np_node
        while lines[i].strip() != "endNpNode":
            node_name = lines[i].strip()

            if node_name not in names_dict:
                raise ValueError("Invalid file format")

            np.append(names_dict[node_name])
            i += 1

        np = sorted(np)

    return (n, nf, nv, F, varFInt, cij, perturbation, np)
