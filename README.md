# BANG - BooleAn Networks on the GPU

BANG is a Python package dedicated to analysis, simulation, and control of Boolean networks with the help of CUDA.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)

## Features

- **High-performance simulation**: GPU-accelerated computations for Boolean network simulations.
- **Effective block-based attractor detection (PBN)**: Efficiently find attractors in large networks.
- **Visualization**: Built-in support for visualizing network dynamics.
- **Parsing**: Easy-to-use parsing of PBNs from ASSA-PBN and SBML formats.

## Installation

BANG is available on PyPI and can be installed using pip. Ensure you have Python 3.10 or higher.

```bash
pip install bang-gpu
```

For the latest development version, you can clone the repository and install it manually. First you need to clone the repository:

```bash
git clone https://github.com/zpp20/bang.git
cd bang
```

Now, you can install the package:

```bash
pip install .
```

## Usage

### 1. Creating a Boolean Network

You can define a Boolean network using a set of parameters of a Probabilistic Boolean Network. For example:

```python
from bang import PBN

pbn = PBN(
    # number of nodes
    3, 
    # number of functions
    [1, 1, 1],
    # number of parents in functions
    [2, 2, 1], 
    # functions
    [[True, True, True, False], [False, True, False, False], [True, False]],
    # indexes of the parents in the functions
    [[0, 1], [0, 1], [1]],
    # probabilities that the functions are chosen
    [[1.0], [1.0], [1.0]],
    # probability of a random perturbation at any step
    0.0,
    # initial state
    [3],
)
```

### 2. Simulating how the network evolves

Run a simulation for a specified number of steps:

```python
# Set the initial state (otherwise the default is all False)
pbn.set_initial_state([[True, False, True]])

# Simulate the network for 10 steps
pbn.simple_steps(10)

# Access the simulation history
print("Trajectory history:", pbn.history)
# Access the final state
print("Last state:", pbn.last_state)
```

### 3. Using CPU-Based simulation

If you want to run the simulation on the CPU instead of the GPU:

```python
# Simulate the network on the CPU for 10 steps
pbn.simple_steps(n_steps=10, device="cpu")
```

### 4. Detecting attractors

To detect attractors in the network, you can use the `attractor_detection` method:

```python
# Detect attractors in the network using the block-based method
attractors = pbn.blocks_detect_attractors()
print("Attractors found:", attractors)
```

### 5. Visualization

You can visualize the state transitions of the network:

```python
# Plot the evolution of the trajectory simulated in the 1st thread
pbn.trajectory_graph(1)
```

## Development

### Setup

The easiest way to set up all the necessary development dependencies is to use `pip` with the `dev` extras:

```bash
pip install -e ".[dev]"
```

### Code quality

BANG uses `black` for code formatting and `isort` for import sorting. It also uses `ruff`  To ensure code quality, you can run the following commands:

```bash
# Format the code
black .
isort .
```

### Testing

BANG uses `pytest` for testing. To run the tests, you can use the following command:

```bash
pytest tests/
```

### Linting

BANG uses `ruff` for linting. To run the linter, you can use the following command:

```bash
ruff check .
```

### Release process

In order to release a new version of BANG, you need to follow these steps:

1. Update the version in `pyproject.toml`.

2. Commit the changes with a message like "Bump version to X.Y.Z".

3. Tag the commit with the new version number: `git tag vX.Y.Z`. Beware that the tag must be in the format `vX.Y.Z[-prerelease]` (e.g., `v1.0.0`).

4. Push the changes and the tag to the remote repository: `git push origin main --tags`.

- If the tag indicates a pre-release version and is not on the main branch, CD will deploy the package to TestPyPI.

- If the commit is on main and the tag does not contain a prerelease version, it will be deployed to PyPI.
