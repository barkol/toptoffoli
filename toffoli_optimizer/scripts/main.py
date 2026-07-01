#!/usr/bin/env python3
"""
Toffoli Optimizer Main Script

This is the entry point for the Toffoli Optimizer project, providing command line
functionality for optimizing quantum circuits.

The script uses object-oriented design with two main classes:
- Optimize: For optimizing a single quantum circuit
"""

import sys
import argparse
import os
import time
import gc
import traceback

# Import our classes
from toffoli_optimizer.scripts.optimize import Optimize

def main():
    """Main function when script is run directly"""
    parser = argparse.ArgumentParser(description='Toffoli Depth Optimizer')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Define a function to add common arguments to all parsers
    def add_common_arguments(parser):
        # We'll add input separately for each subparser with appropriate required setting
        parser.add_argument('--output_dir', type=str, default='toffoli_results',
                           help='Output directory for results')
        parser.add_argument('--topology', type=str, default='linear',
                           choices=['linear', 'grid', 'falcon'], help='Target topology')
        parser.add_argument('--num_qubits', type=int, default=8,
                           help='Number of qubits (overrides the value from input file if specified)')
        parser.add_argument('--num_gates', type=int, default=10,
                           help='Number of gates for generated test circuits')
        parser.add_argument('--fidelity', type=float, default=0.95,
                           help='Target fidelity for optimization')
        parser.add_argument('--min_fidelity', type=float, default=0.85,
                           help='Minimum acceptable fidelity (for aggressive optimization)')
        parser.add_argument('--max_passes', type=int, default=2,
                           help='Maximum number of optimization passes (default: 2)')
        parser.add_argument('--strategy', type=str, default='STANDARD',
                           choices=['STANDARD', 'DEPTH_REDUCTION', 'GATE_REDUCTION', 'FIDELITY',
                                    'HYBRID', 'ULTRA_DEPTH_REDUCTION', 'DEPTH_FIDELITY_BALANCE',
                                    'TRANSPILER', 'TRANSPILER_L1', 'TRANSPILER_L2', 'TRANSPILER_L3'],
                           help='Optimization strategy to use')
        parser.add_argument('--depth_weight', type=float, default=0.7,
                           help='Weight for depth in DEPTH_FIDELITY_BALANCE strategy (default: 0.7)')
        parser.add_argument('--fidelity_weight', type=float, default=0.3,
                           help='Weight for fidelity in DEPTH_FIDELITY_BALANCE strategy (default: 0.3)')
        parser.add_argument('--use_zx', action='store_true',
                           help='Use ZX-calculus optimization if available')
        parser.add_argument('--use_rl', action='store_true',
                           help='Use RL-based optimization if available')
        parser.add_argument('--use_patterns', action='store_true',
                           help='Use pattern-based optimization if available')
        parser.add_argument('--visualize', action='store_true',
                           help='Generate visualization of results')
        parser.add_argument('--verify-equivalence', dest='verify_equivalence',
                           action='store_true',
                           help='Enable the exact equivalence correctness gate: reject any '
                                'rewrite proven NOT function-equivalent to the input '
                                '(skipped for circuits too large to verify exactly). '
                                'Default: off.')
        parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')
        return parser
    
    # Optimizer command (adds a few specific arguments)
    optimize_parser = subparsers.add_parser('optimize', help='Optimize a single Toffoli network')
    optimize_parser = add_common_arguments(optimize_parser)
    # Add input argument with required=True for the optimize command
    optimize_parser.add_argument('--input', type=str, required=True,
                               help='Input file containing the Toffoli network (without extension)')
    optimize_parser.add_argument('--output', type=str, default='optimized_circuit.qasm',
                               help='Output file for the optimized circuit (default: optimized_circuit.qasm)')

    # reduce-count command: minimise the NUMBER of atomic CCX/MCX gates (does not
    # decompose Toffolis), every rewrite verified function-preserving.
    reduce_parser = subparsers.add_parser(
        'reduce-count',
        help='Reduce the number of atomic Toffoli (CCX/MCX) gates, verifier-gated')
    reduce_parser.add_argument('--input', type=str, required=True,
                               help='Input file containing the Toffoli network (without extension)')
    reduce_parser.add_argument('--output', type=str, default='reduced_circuit.qasm',
                               help='Output QASM file (default: reduced_circuit.qasm)')
    reduce_parser.add_argument('--num_qubits', type=int, default=None,
                               help='Override the number of qubits (default: from the input)')
    reduce_parser.add_argument('--allow-permutation', dest='allow_permutation',
                               action='store_true',
                               help='Accept count-reducing rewrites valid up to an output-wire '
                                    'permutation; the permutation is tracked and reported '
                                    '(apply it at read-out). Default: off (strict equivalence).')
    reduce_parser.add_argument('--no-expand-to-cancel', dest='no_expand_to_cancel',
                               action='store_true',
                               help='Disable the expand-to-cancel peephole pass.')
    reduce_parser.add_argument('--no-fanout-cse', dest='no_fanout_cse',
                               action='store_true',
                               help='Disable equal-control compute-once fan-out (CSE).')

    # Parse arguments
    args = parser.parse_args()
    
    # Handle the case when no command is specified
    if args.command is None:
        parser.print_help()
        return 0
    
    # Execute the chosen command using the appropriate class
    if args.command == 'optimize':
        # Wire the opt-in correctness gate. The Optimize wrapper / optimize.py
        # construct ToffoliDepthOptimizer without forwarding this flag, so we flip
        # the class-level default before any optimizer instance is created. This
        # leaves default behavior unchanged when --verify-equivalence is absent.
        if getattr(args, 'verify_equivalence', False):
            from toffoli_optimizer.core.toffoli_depth_optimizer import ToffoliDepthOptimizer
            ToffoliDepthOptimizer._verify_equivalence_default = True
            print("Exact equivalence correctness gate ENABLED (--verify-equivalence)")
        optimizer = Optimize(args)
        return optimizer.run()
    elif args.command == 'reduce-count':
        from toffoli_optimizer.scripts.reduce_count import ReduceCount
        return ReduceCount(args).run()
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())
