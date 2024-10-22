# BANG - BooleAn Networks on the GPU

This is a Python package dedicated to generation, analysis, simulation and control of boolean networks with the help of CUDA.

## In this README :point_down:

- [Features](#features)
- [Initial setup](#initial-setup)
- [Usage](#usage)
- [FAQ](#faq)

## Features

## Initial setup

## Usage

1. First, input "import bang" in your Python script. 


2. Then, create a new instance of the bang.BooleanNetwork class. The simplest way
is to pass the path to the sbml file with the network description. You can also change network_description 
attribute manually later.


3. Granted that the network description is parsed correctly, you can now use 
functions such as simulate, analyze, control, etc.

    ```python
   import bang
   .
   .
   .
   network = bang.BooleanNetwork("path/to/sbml/file")
   results = network.simulate(10)

## FAQ
