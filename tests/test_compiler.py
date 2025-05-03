# tests/test_compiler.py
"""
Unit tests for the ToffoliCompiler class.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit

from toffoli_optimizer.core.compiler import ToffoliCompiler, ToffoliType

class TestToffoliCompiler(unittest.TestCase):
    """Test suite for ToffoliCompiler class"""
    
    def setUp(self):
        """Setup test environment before each test"""
        self.compiler = ToffoliCompiler(debug_mode=False)
    
    def test_init(self):
        """Test compiler initialization"""
        self.assertIsNotNone(self.compiler)
        self.assertEqual(self.compiler.default_basis_gates, ['id', 'rz', 'sx', 'x', 'cx'])
        self.assertFalse(self.compiler.debug_mode)
        self.assertEqual(self.compiler.optimization_level, 3)
        self.assertIsNotNone(self.compiler.coupling_map)
    
    def test_create_simple_toffoli_network(self):
        """Test creating a simple Toffoli network"""
        # Define a simple 2-gate Toffoli network
        toffoli_gates = [
            ([0, 1], 2),  # Controls: 0,1; Target: 2
            ([1, 2], 3),  # Controls: 1,2; Target: 3
        ]
        num_qubits = 4
        
        # Create the network
        circuit, ancilla_indices = self.compiler.create_toffoli_network(
            toffoli_gates,
            num_qubits,
            use_ancilla=False  # No ancilla for simplicity
        )
        
        # Verify the circuit was created
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.num_qubits, num_qubits)
        
        # Verify no ancilla qubits were used
        self.assertIsNone(ancilla_indices)
        
        # Verify circuit depth is non-zero (exact depth depends on implementation)
        self.assertGreater(circuit.depth(), 0)
    
    def test_get_circuit_metrics(self):
        """Test getting circuit metrics"""
        # Create a simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        
        # Get metrics
        metrics = self.compiler.get_circuit_metrics(qc)
        
        # Verify basic metrics
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["width"], 2)
        self.assertEqual(metrics["depth"], 3)
        self.assertEqual(metrics["size"], 3)
        self.assertEqual(metrics["cx_count"], 1)
        
        # Verify gate counts
        self.assertIn("gate_counts", metrics)
        self.assertEqual(metrics["gate_counts"].get("h", 0), 2)
        self.assertEqual(metrics["gate_counts"].get("cx", 0), 1)
    
    def test_estimate_fidelity(self):
        """Test fidelity estimation"""
        # Test with explicit parameters
        fidelity = self.compiler.estimate_fidelity(
            num_qubits=5,
            num_operations=20,
            cx_count=5,
            t_count=2,
            depth=10
        )
        
        # Fidelity should be a float between 0 and 1
        self.assertIsInstance(fidelity, float)
        self.assertGreaterEqual(fidelity, 0.0)
        self.assertLessEqual(fidelity, 1.0)
        
        # Fidelity should decrease with more operations/depth
        fidelity2 = self.compiler.estimate_fidelity(
            num_qubits=5,
            num_operations=40,  # Double the operations
            cx_count=10,
            t_count=4,
            depth=20
        )
        
        # Second fidelity should be lower due to more operations
        self.assertLess(fidelity2, fidelity)
    
    def test_create_toffoli_different_implementations(self):
        """Test different Toffoli implementations"""
        # Test each implementation type with appropriate ancilla
        for toffoli_type in [ToffoliType.STANDARD, ToffoliType.RELATIVE_PHASE_1]:
            # Create a circuit
            qc = QuantumCircuit(6)  # Enough for main qubits + ancilla
            
            # Add a Toffoli gate with the specified implementation
            if toffoli_type == ToffoliType.STANDARD:
                # Standard implementation needs no ancilla
                self.compiler.create_toffoli(qc, 0, 1, 2, toffoli_type)
            else:
                # Relative phase implementation needs 1 ancilla
                self.compiler.create_toffoli(qc, 0, 1, 2, toffoli_type, [3])
            
            # Verify the circuit has non-zero depth
            self.assertGreater(qc.depth(), 0)
            
            # Verify the circuit has gates
            self.assertGreater(len(qc.data), 0)

# tests/test_optimizer.py
"""
Unit tests for the ToffoliDepthOptimizer class.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit

from toffoli_optimizer.core.optimizer import ToffoliDepthOptimizer, OptimizationStrategy
from toffoli_optimizer.core.compiler import ToffoliCompiler, ToffoliType

class TestToffoliDepthOptimizer(unittest.TestCase):
    """Test suite for ToffoliDepthOptimizer class"""
    
    def setUp(self):
        """Setup test environment before each test"""
        self.compiler = ToffoliCompiler(debug_mode=False)
        self.optimizer = ToffoliDepthOptimizer(
            target_fidelity=0.95,
            max_passes=1,  # Single pass for faster tests
            debug_mode=False
        )
        self.optimizer.compiler = self.compiler
    
    def test_init(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.target_fidelity, 0.95)
        self.assertEqual(self.optimizer.max_passes, 1)
        self.assertFalse(self.optimizer.debug_mode)
    
    def test_optimize_simple_network(self):
        """Test optimizing a simple Toffoli network"""
        # Define a simple Toffoli network
        toffoli_gates = [
            ([0, 1], 2),  # Controls: 0,1; Target: 2
            ([1, 2], 3),  # Controls: 1,2; Target: 3
        ]
        num_qubits = 4
        output_qubits = [2, 3]
        input_qubits = [0, 1]
        
        # Run optimization
        results = self.optimizer.optimize_toffoli_network(
            toffoli_gates,
            output_qubits,
            input_qubits,
            num_qubits,
            topology='linear'
        )
        
        # Verify results structure
        self.assertIsNotNone(results)
        self.assertIn("logical", results)
        self.assertIn("optimized", results)
        self.assertIn("mapped", results)
        
        # Verify circuits were created
        self.assertIn("circuit", results["logical"])
        self.assertIn("circuit", results["optimized"])
        self.assertIn("circuit", results["mapped"])
        
        # Verify depths were calculated
        self.assertIn("depth", results["logical"])
        self.assertIn("depth", results["optimized"])
        self.assertIn("depth", results["mapped"])
        
        # Verify depth reduction was calculated
        self.assertIn("depth_reduction", results)
        
        # Verify optimization completed
        self.assertNotIn("error", results)
    
    def test_optimization_strategies(self):
        """Test different optimization strategies"""
        # Define a simple Toffoli network
        toffoli_gates = [
            ([0, 1], 2),  # Controls: 0,1; Target: 2
            ([1, 2], 3),  # Controls: 1,2; Target: 3
        ]
        num_qubits = 4
        output_qubits = [2, 3]
        input_qubits = [0, 1]
        
        # Test each strategy
        for strategy in [
            OptimizationStrategy.STANDARD,
            OptimizationStrategy.DEPTH_REDUCTION,
            OptimizationStrategy.GATE_REDUCTION,
            OptimizationStrategy.FIDELITY,
            OptimizationStrategy.HYBRID
        ]:
            # Create optimizer with this strategy
            optimizer = ToffoliDepthOptimizer(
                target_fidelity=0.95,
                max_passes=1,  # Single pass for faster tests
                debug_mode=False,
                strategy=strategy
            )
            optimizer.compiler = self.compiler
            
            # Run optimization
            results = optimizer.optimize_toffoli_network(
                toffoli_gates,
                output_qubits,
                input_qubits,
                num_qubits,
                topology='linear'
            )
            
            # Verify successful optimization
            self.assertNotIn("error", results)
            self.assertIn("optimized", results)
            self.assertIn("depth", results["optimized"])
    
    def test_estimate_physical_fidelity(self):
        """Test physical fidelity estimation"""
        # Create a simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        
        # Estimate fidelity
        fidelity = self.optimizer.estimate_physical_fidelity(qc)
        
        # Verify fidelity is a float between 0 and 1
        self.assertIsInstance(fidelity, float)
        self.assertGreaterEqual(fidelity, 0.0)
        self.assertLessEqual(fidelity, 1.0)
    
    def test_calculate_logical_fidelity(self):
        """Test logical fidelity calculation"""
        # Create two similar circuits
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.h(1)
        
        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)
        
        # Calculate fidelity
        fidelity = self.optimizer.calculate_logical_fidelity(qc1, qc2)
        
        # Verify fidelity is a float between 0 and 1
        self.assertIsInstance(fidelity, float)
        self.assertGreaterEqual(fidelity, 0.0)
        self.assertLessEqual(fidelity, 1.0)

# tests/run_tests.py
"""
Main test runner for Toffoli Optimizer package.
"""

import unittest
import sys
import os

def run_all_tests():
    """Run all test suites"""
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__))
    
    # Run tests with verbose output
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())