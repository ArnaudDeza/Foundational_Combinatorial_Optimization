from typing import Tuple
import numpy as np


def load_qap_instance(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a Quadratic Assignment Problem (QAP) instance from a QAPLIB‐style file.

    The file format is:
      • First non‐blank, non‐comment line: integer n (problem size).
      • Next n lines: the "flow" matrix A, given as n×n whitespace‐delimited numbers.
      • Next n lines: the "distance" matrix B, similarly n×n numbers.
    Lines may contain extra spaces, blank lines, or comment lines starting with '#' or '//'.
    Matrices may also be wrapped across multiple physical lines (e.g., 10 numbers per line).

    Returns:
        A, B: two numpy arrays of shape (n, n), dtype=float.

    Raises:
        ValueError: if the file is malformed or does not contain exactly n² entries
                    for each matrix.
    """
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        raise ValueError(f"Could not read file '{path}': {e}")
    
    # Helper function to check if a line is a comment or blank
    def is_comment_or_blank(line: str) -> bool:
        stripped = line.strip()
        return (len(stripped) == 0 or 
                stripped.startswith('#') or 
                stripped.startswith('//') or 
                stripped.startswith('!'))
    
    # Helper function to extract numbers from lines
    def extract_numbers(lines_iter, start_idx: int, expected_count: int, matrix_name: str) -> np.ndarray:
        numbers = []
        line_idx = start_idx
        
        while len(numbers) < expected_count and line_idx < len(lines):
            line = lines[line_idx].strip()
            line_idx += 1
            
            # Skip comment or blank lines
            if is_comment_or_blank(line):
                continue
                
            # Split line into tokens and try to convert to float
            tokens = line.split()
            for token in tokens:
                try:
                    numbers.append(float(token))
                except ValueError:
                    raise ValueError(f"Invalid number '{token}' found in {matrix_name} matrix at line {line_idx}")
                    
                # Stop if we have enough numbers
                if len(numbers) == expected_count:
                    break
        
        if len(numbers) < expected_count:
            raise ValueError(f"Expected {expected_count} entries for {matrix_name} matrix, but found only {len(numbers)}")
        elif len(numbers) > expected_count:
            raise ValueError(f"Found {len(numbers)} entries for {matrix_name} matrix, but expected exactly {expected_count}")
            
        return np.array(numbers, dtype=float), line_idx
    
    # Find the problem size n
    n = None
    start_line_idx = 0
    
    for i, line in enumerate(lines):
        if not is_comment_or_blank(line):
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            try:
                n = int(tokens[0])
                start_line_idx = i + 1
                break
            except ValueError:
                raise ValueError(f"Expected problem size (integer) on line {i + 1}, but found '{line.strip()}'")
    
    if n is None:
        raise ValueError("Could not find problem size n in the file")
    
    if n <= 0:
        raise ValueError(f"Problem size must be positive, but found n = {n}")
    
    if n > 10000:  # Sanity check for extremely large problems
        raise ValueError(f"Problem size n = {n} seems unreasonably large")
    
    expected_entries = n * n
    
    # Read matrix A (flow matrix)
    try:
        a_numbers, next_line_idx = extract_numbers(lines, start_line_idx, expected_entries, "flow (A)")
        A = a_numbers.reshape(n, n)
    except Exception as e:
        raise ValueError(f"Error reading flow matrix A: {e}")
    
    # Read matrix B (distance matrix)
    try:
        b_numbers, _ = extract_numbers(lines, next_line_idx, expected_entries, "distance (B)")
        B = b_numbers.reshape(n, n)
    except Exception as e:
        raise ValueError(f"Error reading distance matrix B: {e}")
    
    return A, B


def load_qap_as_symmetric(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a QAP instance and ensure both matrices are symmetric.
    
    This is useful for some QAP variants where the matrices should be symmetric.
    If a matrix is not symmetric, it will be symmetrized by: (M + M.T) / 2
    
    Returns:
        A, B: two symmetric numpy arrays of shape (n, n), dtype=float.
    """
    A, B = load_qap_instance(path)
    
    # Check and fix symmetry if needed
    def make_symmetric(M: np.ndarray, name: str) -> np.ndarray:
        if not np.allclose(M, M.T, atol=1e-10):
            print(f"Warning: {name} matrix is not symmetric. Symmetrizing by (M + M.T) / 2")
            return (M + M.T) / 2
        return M
    
    A_sym = make_symmetric(A, "Flow")
    B_sym = make_symmetric(B, "Distance")
    
    return A_sym, B_sym


if __name__ == "__main__":
    # Simple test - this would need an actual file to work
    try:
        A, B = load_qap_instance("/Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/data/qap/benchmarks/bur26a.dat")
        print("Loaded QAP of size", A.shape[0])
        print("Flow matrix A:")
        print(A)
        print("Distance matrix B:")
        print(B)
    except ValueError as e:
        print(f"Test failed: {e}")
        print("Note: This test requires an actual QAPLIB file at 'path/to/sample.dat'")
     