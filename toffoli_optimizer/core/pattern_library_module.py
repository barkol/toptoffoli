"""
Toffoli Pattern Library Module

This module provides the ToffoliPatternLibrary class for identifying and
simplifying common Toffoli gate patterns to reduce circuit depth.
"""

import itertools

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit, transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Error: Qiskit is required for the pattern library")

class ToffoliPatternLibrary:
    """Library for identifying and simplifying Toffoli gate patterns."""
    
    def __init__(self, debug_mode=False, require_verified=True):
        """Initialize the Toffoli pattern library.

        Args:
            debug_mode: Print extra diagnostics.
            require_verified: When True (default), only simplifications that have
                been PROVEN functionally equivalent to the original Toffoli
                pattern (via ExactEquivalenceVerifier) are ever exposed or
                applied. Non-equivalent simplifications (e.g. relative-phase
                ancilla variants that are only valid in a compute-uncompute
                context) are quarantined and never substituted. Set to False to
                restore the legacy, unchecked behavior.
        """
        from toffoli_optimizer._legacy import warn_legacy
        warn_legacy('ToffoliPatternLibrary')
        self.patterns = {}
        self.simplified_circuits = {}
        self.simplification_metrics = {}
        self.debug_mode = debug_mode
        self.require_verified = require_verified
        # self.verified[pattern_id][method] = bool (True == proven equivalent)
        self.verified = {}
        self.verified_count = 0
        self.quarantined_count = 0

        if QISKIT_AVAILABLE:
            self.build_patterns()
        else:
            print("Warning: Qiskit not available, pattern library will not be built")

    def build_patterns(self):
        """Build the library of Toffoli patterns and their simplifications."""
        # Reset state so a repeated build_patterns() call is idempotent
        # (previously a second call appended duplicate patterns).
        self.patterns = {}
        self.simplified_circuits = {}
        self.simplification_metrics = {}
        self.verified = {}
        self.verified_count = 0
        self.quarantined_count = 0
        # Generate all possible 2-3 Toffoli gate combinations
        self._generate_toffoli_combinations()
        # Simplify each pattern
        self._simplify_patterns()
        # Prove (or disprove) functional equivalence of every simplification.
        self._verify_simplifications()
        # Analyze and sort by simplification potential
        self._analyze_simplifications()
    
    def _generate_toffoli_combinations(self):
        """Generate all possible ways to apply 2-3 Toffoli gates on 3 qubits."""
        # For each Toffoli gate, we need to decide control and target qubits
        # On 3 qubits, we have these possible Toffoli configurations:
        toffoli_configs = [
            ((0, 1), 2),  # Controls: 0,1; Target: 2
            ((0, 2), 1),  # Controls: 0,2; Target: 1
            ((1, 2), 0),  # Controls: 1,2; Target: 0
        ]
        
        # Generate all 2-gate combinations
        for combo in itertools.product(toffoli_configs, repeat=2):
            circuit_key = f"tof2_{len(self.patterns)}"
            circuit = QuantumCircuit(3)
            
            for (controls, target) in combo:
                circuit.ccx(*controls, target)
            
            self.patterns[circuit_key] = {
                "circuit": circuit,
                "sequence": combo,
                "num_gates": 2
            }
        
        # Generate all 3-gate combinations
        for combo in itertools.product(toffoli_configs, repeat=3):
            circuit_key = f"tof3_{len(self.patterns) - 9}"  # Subtract the 9 2-gate patterns
            circuit = QuantumCircuit(3)
            
            for (controls, target) in combo:
                circuit.ccx(*controls, target)
            
            self.patterns[circuit_key] = {
                "circuit": circuit,
                "sequence": combo,
                "num_gates": 3
            }
        
        if self.debug_mode:
            print(f"Generated {len(self.patterns)} Toffoli gate combinations")
    
    def _simplify_patterns(self):
        """Simplify each Toffoli pattern using various techniques."""
        for pattern_id, pattern in self.patterns.items():
            # Create three different simplified versions:
            original_circuit = pattern["circuit"]
            
            # 1. Standard decomposition with no ancillas
            no_ancilla_circuit = self._simplify_no_ancilla(original_circuit)
            
            # 2. Decomposition with 1 ancilla
            one_ancilla_circuit = self._simplify_with_one_ancilla(original_circuit)
            
            # 3. Optimized with multiple ancillas
            multi_ancilla_circuit = self._simplify_with_multi_ancilla(original_circuit)
            
            self.simplified_circuits[pattern_id] = {
                "no_ancilla": no_ancilla_circuit,
                "one_ancilla": one_ancilla_circuit,
                "multi_ancilla": multi_ancilla_circuit
            }

    def _verify_simplifications(self):
        """Prove functional equivalence of each (pattern_id, method) simplification.

        Uses the committed ExactEquivalenceVerifier to compare every simplified
        circuit against the original Toffoli pattern circuit (handling ancilla
        qubits and global phase). Results are recorded in self.verified so the
        application path can quarantine non-equivalent rewrites.
        """
        from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier

        verifier = ExactEquivalenceVerifier()
        self.verified = {}
        self.verified_count = 0
        self.quarantined_count = 0

        for pattern_id, pattern in self.patterns.items():
            original_circuit = pattern["circuit"]
            methods = self.simplified_circuits.get(pattern_id, {})
            self.verified[pattern_id] = {}

            for method, simplified_circuit in methods.items():
                is_equivalent = False
                if simplified_circuit is not None:
                    try:
                        is_equivalent, _perm, _info = verifier.verify(
                            original_circuit, simplified_circuit, allow_permutation=False
                        )
                    except Exception as ex:
                        is_equivalent = False
                        if self.debug_mode:
                            print(f"Verification error for {pattern_id}/{method}: {ex}")

                self.verified[pattern_id][method] = bool(is_equivalent)
                if is_equivalent:
                    self.verified_count += 1
                else:
                    self.quarantined_count += 1

        if self.debug_mode:
            print(f"Pattern verification: {self.verified_count} verified, "
                  f"{self.quarantined_count} quarantined")

    def is_verified(self, pattern_id, method):
        """Return True iff (pattern_id, method) was proven functionally equivalent."""
        return bool(self.verified.get(pattern_id, {}).get(method, False))

    def verification_report(self):
        """Return a dict summarizing how many simplifications verified vs were quarantined."""
        return {
            "verified": self.verified_count,
            "quarantined": self.quarantined_count,
            "total": self.verified_count + self.quarantined_count,
            "require_verified": self.require_verified,
        }

    def print_verification_report(self):
        """Print a small report of verified vs quarantined simplifications."""
        report = self.verification_report()
        print("\nToffoli pattern verification report:")
        print(f"  total simplifications : {report['total']}")
        print(f"  verified equivalent   : {report['verified']}")
        print(f"  quarantined (unsafe)  : {report['quarantined']}")
        print(f"  require_verified      : {report['require_verified']} "
              f"({'only verified simplifications are applied' if report['require_verified'] else 'legacy: unchecked'})")
        return report

    def _simplify_no_ancilla(self, circuit):
        """Simplify a Toffoli circuit without using ancilla qubits."""
        # Use Qiskit's built-in transpiler with optimization level 3
        simplified = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=1)
        return simplified
    
    def _simplify_with_one_ancilla(self, circuit):
        """Simplify a Toffoli circuit using one ancilla qubit."""
        # Create a new circuit with an ancilla
        num_qubits = circuit.num_qubits
        circuit_with_ancilla = QuantumCircuit(num_qubits + 1)
        
        # Copy the original circuit
        for instruction in circuit.data:
            if instruction.operation.name == 'ccx':
                # Replace CCX with a relative-phase Toffoli using the ancilla
                # Qiskit 2.0 compatible
                controls = []
                for control_qubit in [instruction.qubits[0], instruction.qubits[1]]:
                    if hasattr(control_qubit, 'index'):
                        controls.append(control_qubit.index)
                    elif hasattr(control_qubit, '_index'):
                        controls.append(control_qubit._index)
                    else:
                        # Try string parsing as last resort
                        str_rep = str(control_qubit)
                        if 'q[' in str_rep and ']' in str_rep:
                            idx_str = str_rep.split('q[')[1].split(']')[0]
                            controls.append(int(idx_str))
                
                target_qubit = instruction.qubits[2]
                if hasattr(target_qubit, 'index'):
                    target = target_qubit.index
                elif hasattr(target_qubit, '_index'):
                    target = target_qubit._index
                else:
                    # Try string parsing as last resort
                    str_rep = str(target_qubit)
                    if 'q[' in str_rep and ']' in str_rep:
                        idx_str = str_rep.split('q[')[1].split(']')[0]
                        target = int(idx_str)
                ancilla = num_qubits  # The added ancilla qubit
                
                # Implement relative-phase Toffoli with one ancilla
                circuit_with_ancilla.h(target)
                circuit_with_ancilla.cx(controls[0], ancilla)
                circuit_with_ancilla.cx(controls[1], ancilla)
                circuit_with_ancilla.cx(ancilla, target)
                circuit_with_ancilla.cx(controls[0], ancilla)
                circuit_with_ancilla.cx(controls[1], ancilla)
                circuit_with_ancilla.h(target)
            else:
                # Copy other gates directly
                qubits = [q.index if hasattr(q, 'index') else q._index for q in instruction.qubits]
                circuit_with_ancilla.append(instruction.operation, qubits)
        
        # Optimize the circuit further
        simplified = transpile(circuit_with_ancilla, basis_gates=['u', 'cx'], optimization_level=1)
        return simplified
    
    def _simplify_with_multi_ancilla(self, circuit):
        """Simplify a Toffoli circuit using multiple ancilla qubits."""
        # Create a new circuit with multiple ancillas (one per Toffoli)
        num_qubits = circuit.num_qubits
        toffoli_count = sum(1 for inst in circuit.data if inst.operation.name == 'ccx')
        circuit_with_ancillas = QuantumCircuit(num_qubits + toffoli_count)
        
        # Count for ancilla allocation
        ancilla_idx = num_qubits
        
        # Copy the original circuit
        for instruction in circuit.data:
            if instruction.operation.name == 'ccx':
                # Replace CCX with an optimized version using one ancilla per Toffoli
                controls = [
                    instruction.qubits[0]._index if hasattr(instruction.qubits[0], '_index') else instruction.qubits[0].index,
                    instruction.qubits[1]._index if hasattr(instruction.qubits[1], '_index') else instruction.qubits[1].index
                ]
                target = instruction.qubits[2]._index if hasattr(instruction.qubits[2], '_index') else instruction.qubits[2].index
                ancilla = ancilla_idx
                ancilla_idx += 1
                
                # Implement optimized Toffoli with one dedicated ancilla
                # This uses the relative-phase Toffoli implementation
                circuit_with_ancillas.h(target)
                circuit_with_ancillas.cx(controls[0], ancilla)
                circuit_with_ancillas.cx(controls[1], ancilla)
                circuit_with_ancillas.cx(ancilla, target)
                circuit_with_ancillas.cx(controls[0], ancilla)
                circuit_with_ancillas.cx(controls[1], ancilla)
                circuit_with_ancillas.h(target)
            else:
                # Copy other gates directly
                qubits = [q.index if hasattr(q, 'index') else q._index for q in instruction.qubits]
                circuit_with_ancillas.append(instruction.operation, qubits)
        
        # Optimize the circuit further
        simplified = transpile(circuit_with_ancillas, basis_gates=['u', 'cx'], optimization_level=1)
        return simplified
    
    def _analyze_simplifications(self):
        """Analyze the simplifications and identify the best candidates."""
        for pattern_id in self.patterns.keys():
            # Get the original and simplified circuits
            original_circuit = self.patterns[pattern_id]["circuit"]
            no_ancilla_circuit = self.simplified_circuits[pattern_id]["no_ancilla"]
            one_ancilla_circuit = self.simplified_circuits[pattern_id]["one_ancilla"]
            multi_ancilla_circuit = self.simplified_circuits[pattern_id]["multi_ancilla"]
            
            # Calculate metrics
            original_depth = original_circuit.depth()
            original_size = original_circuit.size()
            original_cx_count = sum(1 for inst in original_circuit.data if inst.operation.name == 'cx')
            
            no_ancilla_depth = no_ancilla_circuit.depth()
            no_ancilla_size = no_ancilla_circuit.size()
            no_ancilla_cx_count = sum(1 for inst in no_ancilla_circuit.data if inst.operation.name == 'cx')
            
            one_ancilla_depth = one_ancilla_circuit.depth()
            one_ancilla_size = one_ancilla_circuit.size()
            one_ancilla_cx_count = sum(1 for inst in one_ancilla_circuit.data if inst.operation.name == 'cx')
            
            multi_ancilla_depth = multi_ancilla_circuit.depth()
            multi_ancilla_size = multi_ancilla_circuit.size()
            multi_ancilla_cx_count = sum(1 for inst in multi_ancilla_circuit.data if inst.operation.name == 'cx')
            
            # Calculate improvement percentages (depth reduction)
            if original_depth > 0:
                no_ancilla_depth_reduction = ((original_depth - no_ancilla_depth) / original_depth) * 100
                one_ancilla_depth_reduction = ((original_depth - one_ancilla_depth) / original_depth) * 100
                multi_ancilla_depth_reduction = ((original_depth - multi_ancilla_depth) / original_depth) * 100
            else:
                no_ancilla_depth_reduction = 0
                one_ancilla_depth_reduction = 0
                multi_ancilla_depth_reduction = 0
            
            # Candidate methods, each with its depth-reduction figure. When
            # require_verified is on, only simplifications PROVEN equivalent are
            # eligible to be chosen as the best method (the rest are quarantined
            # and must never be applied).
            method_data = {
                "no_ancilla": (no_ancilla_depth_reduction, no_ancilla_circuit,
                               no_ancilla_depth, no_ancilla_size, no_ancilla_cx_count),
                "one_ancilla": (one_ancilla_depth_reduction, one_ancilla_circuit,
                                one_ancilla_depth, one_ancilla_size, one_ancilla_cx_count),
                "multi_ancilla": (multi_ancilla_depth_reduction, multi_ancilla_circuit,
                                  multi_ancilla_depth, multi_ancilla_size, multi_ancilla_cx_count),
            }

            eligible = [m for m in method_data
                        if (not self.require_verified) or self.is_verified(pattern_id, m)]

            if eligible:
                best_method = max(eligible, key=lambda m: method_data[m][0])
                best_reduction, best_circuit, best_depth, best_size, best_cx_count = method_data[best_method]
            else:
                # No safe simplification: do nothing (identity), so the optimizer
                # leaves this pattern untouched rather than corrupting it.
                best_method = None
                best_reduction = 0
                best_circuit = original_circuit
                best_depth = original_depth
                best_size = original_size
                best_cx_count = original_cx_count
            
            # Store metrics
            self.simplification_metrics[pattern_id] = {
                "original_depth": original_depth,
                "original_size": original_size,
                "original_cx_count": original_cx_count,
                "best_method": best_method,
                "best_depth": best_depth,
                "best_size": best_size,
                "best_cx_count": best_cx_count,
                "depth_reduction": best_reduction,
                "no_ancilla_depth_reduction": no_ancilla_depth_reduction,
                "one_ancilla_depth_reduction": one_ancilla_depth_reduction,
                "multi_ancilla_depth_reduction": multi_ancilla_depth_reduction
            }
    
    def get_top_simplifications(self, n=5):
        """Get the top N patterns with the best simplification potential."""
        # Sort patterns by depth reduction
        sorted_patterns = sorted(
            self.simplification_metrics.items(),
            key=lambda x: x[1]["depth_reduction"],
            reverse=True
        )
        
        return sorted_patterns[:n]
    
    def print_top_simplifications(self, n=5):
        """Print the top N patterns with the best simplification potential."""
        top_patterns = self.get_top_simplifications(n)
        
        print(f"\nTop {n} Toffoli Patterns with Greatest Simplification Potential:\n")
        print("-" * 80)
        
        for i, (pattern_id, metrics) in enumerate(top_patterns):
            pattern_sequence = self.patterns[pattern_id]["sequence"]
            simplified_method = metrics["best_method"]
            
            print(f"{i+1}. Pattern ID: {pattern_id}")
            print(f"   Original sequence: {pattern_sequence}")
            print(f"   Original depth: {metrics['original_depth']}")
            print(f"   Best simplification method: {simplified_method}")
            print(f"   Simplified depth: {metrics['best_depth']}")
            print(f"   Depth reduction: {metrics['depth_reduction']:.2f}%")
            print(f"   CX count reduction: {metrics['original_cx_count']} → {metrics['best_cx_count']}")
            print("-" * 80)
    
    def find_patterns_in_circuit(self, circuit):
        """
        Find all Toffoli patterns in a given circuit.
        
        Args:
            circuit: The circuit to analyze
            
        Returns:
            list: List of (pattern_id, start_index) tuples for all found patterns
        """
        found_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            pattern_length = len(pattern["sequence"])
            
            # Skip if circuit is too small
            if len(circuit.data) < pattern_length:
                continue
            
            # Scan the circuit for this pattern
            for i in range(len(circuit.data) - pattern_length + 1):
                segment = circuit.data[i:i+pattern_length]
                
                # Check if segment matches pattern
                matches = True
                for j, instruction in enumerate(segment):
                    if instruction.operation.name != 'ccx':
                        matches = False
                        break
                    
                    pattern_controls, pattern_target = pattern["sequence"][j]
                    
                    # Get the qubit indices from the instruction
                    instruction_qubits = []
                    for q in instruction.qubits:
                        if hasattr(q, 'index'):
                            instruction_qubits.append(q.index)
                        elif hasattr(q, '_index'):
                            instruction_qubits.append(q._index)
                        else:
                            if self.debug_mode:
                                print(f"Warning: Could not determine qubit index for {q}")
                            matches = False
                            break
                    
                    if not matches:
                        break
                        
                    instruction_controls = instruction_qubits[:2]
                    instruction_target = instruction_qubits[2]
                    
                    # Check if they match
                    if sorted(instruction_controls) != sorted(pattern_controls) or instruction_target != pattern_target:
                        matches = False
                        break
                
                if matches:
                    found_patterns.append((pattern_id, i))
        
        return found_patterns
    
    def optimize_circuit(self, circuit, threshold=10.0, use_ancilla=True):
        """
        Optimize a circuit by replacing Toffoli patterns with simplified versions.
        
        Args:
            circuit: The circuit to optimize
            threshold: Minimum depth reduction percentage to apply a simplification
            use_ancilla: Whether to allow solutions with ancilla qubits
            
        Returns:
            QuantumCircuit: The optimized circuit
        """
        # Find all patterns
        patterns = self.find_patterns_in_circuit(circuit)
        
        if not patterns:
            return circuit
            
        if self.debug_mode:
            print(f"Found {len(patterns)} Toffoli patterns in the circuit")
        
        # Sort by position in reverse order (to handle overlapping patterns properly)
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Create a copy of the circuit to modify
        optimized_circuit = circuit.copy()
        
        # Apply simplifications for patterns that meet the threshold
        replacements = 0
        for pattern_id, start_idx in patterns:
            metrics = self.simplification_metrics[pattern_id]

            if metrics["depth_reduction"] >= threshold:
                # Choose the best method (respecting use_ancilla parameter)
                if use_ancilla:
                    best_method = metrics["best_method"]
                else:
                    best_method = "no_ancilla"

                # If there is no safe method for this pattern, skip it entirely.
                if best_method is None:
                    continue

                # Quarantine guard: never apply a simplification that was not
                # proven functionally equivalent to the original Toffoli pattern.
                if self.require_verified and not self.is_verified(pattern_id, best_method):
                    if self.debug_mode:
                        print(f"Skipping unverified simplification "
                              f"{pattern_id}/{best_method} (quarantined)")
                    continue

                pattern_length = len(self.patterns[pattern_id]["sequence"])
                simplified_circuit = self.simplified_circuits[pattern_id][best_method]
                
                # Extract the pattern from the circuit
                pattern_segment = optimized_circuit.data[start_idx:start_idx+pattern_length]
                
                # Remove the pattern
                for instruction in pattern_segment:
                    optimized_circuit.data.remove(instruction)
                
                # Insert the simplified version
                for i, instruction in enumerate(simplified_circuit.data):
                    optimized_circuit.data.insert(start_idx + i, instruction)
                
                replacements += 1
        
        if self.debug_mode and replacements > 0:
            print(f"Applied {replacements} pattern replacements")
        
        return optimized_circuit
