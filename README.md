# Task Scheduling Optimization

This repository contains the implementation of a quantum-inspired tensor network approach for task scheduling optimization in industrial plants, as described in the paper ["Task Scheduling Optimization from a Tensor Network Perspective"](https://arxiv.org/abs/2311.10433).

## Overview

The project implements a novel method for optimizing task scheduling using quantum-inspired tensor network technology. It finds optimal combinations of tasks on a set of machines while satisfying given constraints, without evaluating all possible combinations. The method simulates a quantum system, performs imaginary time evolution, and applies projections to satisfy constraints.

## Project Structure

- `app.py` - Main Streamlit application for running the optimization
- `optimizer.py` - Core implementation of the tensor network optimization algorithm
- `genetic_algorithm.py` - Implementation of the genetic algorithm for improved scalability
- `auxiliar_functions.py` - Helper functions used across the project
- `rule_process.py` - Implementation of constraint processing and rule handling
- `task_scheduling_tests.ipynb` - Jupyter notebook with tests and examples
- `figures/` - Directory containing performance and scaling plots
- `results/` - Directory storing optimization results in NumPy format

## Key Features

- Quantum-inspired tensor network optimization
- Improved scalability through:
  - Condensation methods
  - Iterative algorithm
  - Genetic algorithm integration
- Direct constraint handling
- Performance visualization and analysis

## Requirements

```
streamlit>=1.24.0
numpy>=1.24.0
tensornetwork>=0.4.6
pandas
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the optimization using either:

1. The Streamlit web interface:
   ```bash
   streamlit run app.py
   ```

2. The Jupyter notebook:
   - Open `task_scheduling_tests.ipynb` in Jupyter
   - Follow the examples and test cases provided

## Results

The `figures/` directory contains various performance plots:
- Scaling analysis with different parameters
- Success rate evaluations
- Performance comparisons
- Constraint satisfaction metrics

Raw numerical results are stored in the `results/` directory as NumPy arrays.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ali2024taskschedulingoptimizationtensor,
      title={Task Scheduling Optimization from a Tensor Network Perspective}, 
      author={Alejandro Mata Ali and Iñigo Perez Delgado and Beatriz García Markaida and Aitor Moreno Fdez. de Leceta},
      year={2024},
      eprint={2311.10433},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2311.10433}, 
}
```

## License

MIT License