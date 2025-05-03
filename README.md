# Toffoli Optimizer

A tool for optimizing Toffoli networks for quantum circuits to reduce depth and respect hardware connectivity constraints.

## Features

- Toffoli network compilation with different implementation strategies
- Advanced depth optimization algorithms
- Pattern-based optimization for common Toffoli combinations
- Hardware-aware optimization respecting coupling constraints
- Comprehensive benchmarking against Qiskit's transpiler

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from toffoli_optimizer.core.optimizer import ToffoliDepthOptimizer
from toffoli_optimizer.core.compiler import ToffoliCompiler, ToffoliType

# Create a compiler and optimizer
compiler = ToffoliCompiler()
optimizer = ToffoliDepthOptimizer(target_fidelity=0.95, max_passes=2)

# Define a Toffoli network
toffoli_gates = [
    ([0, 1], 2),  # Controls: 0,1; Target: 2
    ([2, 3], 4),  # Controls: 2,3; Target: 4
    ([0, 4], 3),  # Controls: 0,4; Target: 3
]
output_qubits = [2, 3, 4]
input_qubits = [0, 1]
num_qubits = 5

# Optimize the network
results = optimizer.optimize_toffoli_network(
    toffoli_gates,
    output_qubits,
    input_qubits,
    num_qubits,
    topology='linear'
)

# Get the optimized circuit
optimized_circuit = results["optimized"]["circuit"]
print(f"Optimized depth: {results['optimized']['depth']}")
print(f"Physical depth: {results['mapped']['depth']}")
print(f"Depth reduction: {results['depth_reduction']:.2f}%")
```

## Benchmark

```bash
python scripts/main.py --network variable --num_gates 10 --num_qubits 12 --topology grid
```

## License

MIT License
