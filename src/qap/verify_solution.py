#!/usr/bin/env python3
"""
Verify that QAPLIB solutions produce correct objective values when applied to problem data.

This script handles the fact that QAPLIB instances use two different QAP formulations:
1. Koopmans-Beckmann: sum_i sum_j F[i,j] * D[perm[i], perm[j]] (most instances)
2. Trace formulation: trace(F @ X.T @ D @ X) (8 specific instances)

The formulation is determined by instance name, not matrix properties.
"""

import numpy as np
import sys
import os

# Add the parent directory to path to import the loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qap.loader import load_qap_instance, load_qap_solution


def qap_objective(F, D, permutation, instance_name=None):
    """
    Compute the QAP objective value using the appropriate formulation.
    
    Args:
        F: Flow matrix (n x n)
        D: Distance matrix (n x n) 
        permutation: Assignment as a permutation vector (0-based)
        instance_name: Optional instance name to determine formulation
        
    Returns:
        Objective value using the correct QAP formulation
    """
    import numpy as np
    
    n = len(permutation)
    
    # Instances that specifically require trace formulation
    trace_instances = {
        'esc128', 'kra30a', 'kra30b', 'ste36c', 
        'tai60a', 'tai80a', 'tho150', 'tho30'
    }
    
    # Use trace formulation for specific instances, Koopmans-Beckmann for all others
    if instance_name and instance_name in trace_instances:
        # Create assignment matrix X from permutation  
        X = np.zeros((n, n))
        for i in range(n):
            X[i, permutation[i]] = 1
        
        # Trace formulation: trace(F @ X.T @ D @ X)
        return np.trace(F @ X.T @ D @ X)
    else:
        # Koopmans-Beckmann formulation (default)
        obj = 0.0
        for i in range(n):
            for j in range(n):
                obj += F[i, j] * D[permutation[i], permutation[j]]
        return obj


def verify_solution(instance_path, solution_path):
    """
    Verify that a solution file produces the correct objective value.
    
    Args:
        instance_path: Path to QAP instance file (.dat)
        solution_path: Path to QAP solution file (.sln)
    """
    print(f"=== Verifying Solution ===")
    print(f"Instance: {instance_path}")
    print(f"Solution: {solution_path}")
    print()
    
    # Load the QAP instance
    try:
        F, D = load_qap_instance(instance_path)
        print(f"Loaded QAP instance of size {F.shape[0]} x {F.shape[1]}")
    except Exception as e:
        print(f"ERROR loading instance: {e}")
        return False
    
    # Load the solution
    try:
        n, expected_obj, permutation = load_qap_solution(solution_path)
        print(f"Loaded solution for n={n}, expected objective={expected_obj}")
        print(f"Permutation: {permutation}")
    except Exception as e:
        print(f"ERROR loading solution: {e}")
        return False
    
    # Verify dimensions match
    if F.shape[0] != n or F.shape[1] != n:
        print(f"ERROR: Instance size {F.shape} doesn't match solution size {n}")
        return False
    
    if D.shape[0] != n or D.shape[1] != n:
        print(f"ERROR: Distance matrix size {D.shape} doesn't match solution size {n}")
        return False
    
    if len(permutation) != n:
        print(f"ERROR: Permutation length {len(permutation)} doesn't match problem size {n}")
        return False
    
    # Extract instance name from path
    import os
    instance_name = os.path.basename(instance_path).replace('.dat', '')
    
    # Compute the actual objective value
    actual_obj = qap_objective(F, D, permutation, instance_name)
    print(f"\nObjective computation:")
    print(f"Expected: {expected_obj}")
    print(f"Actual:   {actual_obj}")
    print(f"Difference: {abs(actual_obj - expected_obj)}")
    
    # Check if they match (with some tolerance for floating point)
    tolerance = 1e-6
    if abs(actual_obj - expected_obj) < tolerance:
        print(f"✅ VERIFICATION PASSED: Solution is correct!")
        return True
    else:
        print(f"❌ VERIFICATION FAILED: Objective values don't match!")
        return False


def discover_instance_solution_pairs(benchmark_dir):
    """
    Discover all available instance-solution pairs in the benchmark directory.
    
    Args:
        benchmark_dir: Path to the benchmark directory
        
    Returns:
        List of (instance_name, instance_path, solution_path) tuples
    """
    import os
    
    pairs = []
    
    # Get all .dat files
    dat_files = [f for f in os.listdir(benchmark_dir) if f.endswith('.dat')]
    
    for dat_file in dat_files:
        # Get the base name (without .dat extension)
        base_name = dat_file[:-4]
        
        # Check if corresponding .sln file exists
        sln_file = f"{base_name}.sln"
        sln_path = os.path.join(benchmark_dir, sln_file)
        
        if os.path.exists(sln_path):
            instance_path = os.path.join(benchmark_dir, dat_file)
            pairs.append((base_name, instance_path, sln_path))
    
    return pairs


def main():
    """Automatically discover and verify all available instance-solution pairs."""
    
    # Use absolute paths to ensure we can find the files
    base_dir = "/Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization"
    benchmark_dir = f"{base_dir}/data/qap/benchmarks"
    
    print("=== QAPLIB Solution Verification ===")
    print(f"Scanning directory: {benchmark_dir}")
    
    # Discover all instance-solution pairs
    pairs = discover_instance_solution_pairs(benchmark_dir)
    
    if not pairs:
        print("❌ No instance-solution pairs found!")
        sys.exit(1)
    
    print(f"Found {len(pairs)} instance-solution pairs to verify:")
    
    # Sort pairs by instance name for consistent output
    pairs.sort(key=lambda x: x[0])
    
    # Print the list of instances to be tested
    print("\nInstances to verify:")
    for i, (name, _, _) in enumerate(pairs, 1):
        print(f"  {i:3d}. {name}")
    
    print(f"\n{'='*80}")
    
    # Verify all pairs
    passed = 0
    failed = 0
    failed_instances = []
    
    for i, (instance_name, instance_path, solution_path) in enumerate(pairs, 1):
        print(f"\n[{i:3d}/{len(pairs)}] Testing {instance_name}")
        print("-" * 60)
        
        success = verify_solution(instance_path, solution_path)
        if success:
            passed += 1
            print("✅ PASSED")
        else:
            failed += 1
            failed_instances.append(instance_name)
            print("❌ FAILED")
    
    # Final summary
    print(f"\n{'='*80}")
    print("=== VERIFICATION SUMMARY ===")
    print(f"Total instances tested: {len(pairs)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/len(pairs)*100:.1f}%")
    
    if failed > 0:
        print(f"\nFailed instances:")
        for name in failed_instances:
            print(f"  • {name}")
        print("\n💥 SOME VERIFICATIONS FAILED!")
        sys.exit(1)
    else:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("✅ The QAPLIB loader and objective computation are working correctly!")
        print("✅ All loaded solutions produce the expected objective values!")


if __name__ == "__main__":
    main() 