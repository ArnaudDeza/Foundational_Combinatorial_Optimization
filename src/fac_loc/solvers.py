#!/usr/bin/env python3
"""
Facility Location Problem Solvers.
Supports both Uncapacitated Facility Location Problem (UFLP) and P-median problems.
"""

import argparse
import logging
import numpy as np
import os
import glob
from pathlib import Path
from abc import ABC, abstractmethod
import pickle

# Optional Gurobi import
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("Warning: Gurobi not available. Solvers will not work without Gurobi license.")

# For parallel solving
from joblib import Parallel, delayed
from tqdm import tqdm

# Optional tqdm_joblib import
try:
    from tqdm_joblib import tqdm_joblib
    HAS_TQDM_JOBLIB = True
except ImportError:
    HAS_TQDM_JOBLIB = False
    def tqdm_joblib(tqdm_object):
        # Simple context manager that does nothing
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()


# =============================================================================
# Base and Solver Classes
# =============================================================================

class FacilityLocationSolver(ABC):
    """Abstract base class for facility location problem solvers."""

    def __init__(self, time_limit=None, max_threads=None, mip_gap=None, mip_focus=None):
        self.time_limit = time_limit
        self.max_threads = max_threads
        self.mip_gap = mip_gap
        self.mip_focus = mip_focus

    def _check_gurobi(self):
        """Check if Gurobi is available."""
        if not HAS_GUROBI:
            raise ImportError("Gurobi is not available. Please install Gurobi and obtain a license.")

    def _configure_model(self, model):
        """Configure Gurobi model parameters."""
        model.setParam("OutputFlag", 0)
        if self.time_limit is not None:
            model.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            model.setParam("Threads", self.max_threads)
        if self.mip_gap is not None:
            model.setParam("MIPGap", self.mip_gap)
        if self.mip_focus is not None:
            model.setParam("MIPFocus", self.mip_focus)

    @abstractmethod
    def solve(self, instance):
        """
        Solve the given facility location problem instance.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class UFLPSolver(FacilityLocationSolver):
    """Solver for the Uncapacitated Facility Location Problem (UFLP)."""

    def solve(self, instance):
        """Solve UFLP instance using Gurobi MIP formulation."""
        self._check_gurobi()
        
        # Extract instance data
        fixed_costs = instance["fixed_costs"]
        transport_cost = instance["transport_cost"]
        n_facilities = len(fixed_costs)
        
        # Validate instance
        if transport_cost.shape != (n_facilities, n_facilities):
            raise ValueError(f"Transport cost matrix shape {transport_cost.shape} "
                           f"doesn't match number of facilities {n_facilities}")

        # Create model
        model = gp.Model("UFLP")
        self._configure_model(model)

        # Decision variables
        # x[i,j] = 1 if client i is assigned to facility j
        x = model.addVars(n_facilities, n_facilities, vtype=GRB.BINARY, name='x')
        # y[j] = 1 if facility j is opened
        y = model.addVars(n_facilities, vtype=GRB.BINARY, name='y')

        # Objective: minimize total transportation and fixed costs
        model.setObjective(
            gp.quicksum(transport_cost[i, j] * x[i, j]
                       for i in range(n_facilities)
                       for j in range(n_facilities)) +
            gp.quicksum(fixed_costs[j] * y[j] for j in range(n_facilities)),
            sense=GRB.MINIMIZE
        )

        # Constraints
        # Each client must be assigned to exactly one facility
        model.addConstrs(
            gp.quicksum(x[i, j] for j in range(n_facilities)) == 1
            for i in range(n_facilities)
        )

        # A client can only be assigned to an open facility
        model.addConstrs(
            x[i, j] <= y[j]
            for i in range(n_facilities) 
            for j in range(n_facilities)
        )

        # Solve
        model.optimize()

        # Extract solution
        result = self._extract_solution(model, instance, x, y, n_facilities)
        return result

    def _extract_solution(self, model, instance, x, y, n_facilities):
        """Extract solution from solved model."""
        # Check if solution exists
        if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
            try:
                sol_y = np.array([y[j].X for j in range(n_facilities)])
                sol_x = np.array([[x[i, j].X for j in range(n_facilities)] 
                                for i in range(n_facilities)])
                obj_val = model.objVal
                bound = model.ObjBound if hasattr(model, 'ObjBound') else obj_val
            except:
                # Fallback if variable access fails
                sol_y = np.zeros(n_facilities)
                sol_x = np.zeros((n_facilities, n_facilities))
                obj_val = float('inf')
                bound = float('inf')
        else:
            # No feasible solution found
            sol_y = np.zeros(n_facilities)
            sol_x = np.zeros((n_facilities, n_facilities))
            obj_val = float('inf')
            bound = float('inf')

        # Build result dictionary
        result = {
            'problem_type': 'uflp',
            'transport_cost': instance.get("transport_cost"),
            'fixed_costs': instance.get("fixed_costs"),
            'nv': instance.get("nv"),
            'nc': instance.get("nc"),
            'seed': instance.get("seed"),
            'n_facilities': n_facilities,
            'mip_status': model.Status,
            'mip_obj_val': obj_val,
            'mip_runtime': model.Runtime,
            'mip_num_nodes': model.getAttr('NodeCount'),
            'mip_gap': model.MIPGap if model.SolCount > 0 else float('inf'),
            'best_bound': bound,
            'sol_facilities': sol_y,
            'sol_assignments': sol_x,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads
        }
        
        return result


class PMedianSolver(FacilityLocationSolver):
    """Solver for the P-median Problem."""

    def solve(self, instance):
        """Solve P-median instance using Gurobi MIP formulation."""
        self._check_gurobi()
        
        # Extract instance data
        fixed_costs = instance["fixed_costs"]
        transport_cost = instance["transport_cost"]
        k = instance["k"]  # Number of facilities to open
        n_facilities = len(fixed_costs)
        
        # Validate instance
        if transport_cost.shape != (n_facilities, n_facilities):
            raise ValueError(f"Transport cost matrix shape {transport_cost.shape} "
                           f"doesn't match number of facilities {n_facilities}")
        if k > n_facilities:
            raise ValueError(f"Cannot open {k} facilities when only {n_facilities} are available")

        # Create model
        model = gp.Model("P-median")
        self._configure_model(model)

        # Decision variables
        # x[i,j] = 1 if client i is assigned to facility j
        x = model.addVars(n_facilities, n_facilities, vtype=GRB.BINARY, name='x')
        # y[j] = 1 if facility j is opened
        y = model.addVars(n_facilities, vtype=GRB.BINARY, name='y')

        # Objective: minimize total transportation costs (no fixed costs in p-median)
        model.setObjective(
            gp.quicksum(transport_cost[i, j] * x[i, j]
                       for i in range(n_facilities)
                       for j in range(n_facilities)),
            sense=GRB.MINIMIZE
        )

        # Constraints
        # Each client must be assigned to exactly one facility
        model.addConstrs(
            gp.quicksum(x[i, j] for j in range(n_facilities)) == 1
            for i in range(n_facilities)
        )

        # A client can only be assigned to an open facility
        model.addConstrs(
            x[i, j] <= y[j]
            for i in range(n_facilities) 
            for j in range(n_facilities)
        )

        # Exactly k facilities must be opened
        model.addConstr(
            gp.quicksum(y[j] for j in range(n_facilities)) == k
        )

        # Solve
        model.optimize()

        # Extract solution
        result = self._extract_solution(model, instance, x, y, n_facilities, k)
        return result

    def _extract_solution(self, model, instance, x, y, n_facilities, k):
        """Extract solution from solved model."""
        # Check if solution exists
        if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
            try:
                sol_y = np.array([y[j].X for j in range(n_facilities)])
                sol_x = np.array([[x[i, j].X for j in range(n_facilities)] 
                                for i in range(n_facilities)])
                obj_val = model.objVal
                bound = model.ObjBound if hasattr(model, 'ObjBound') else obj_val
            except:
                # Fallback if variable access fails
                sol_y = np.zeros(n_facilities)
                sol_x = np.zeros((n_facilities, n_facilities))
                obj_val = float('inf')
                bound = float('inf')
        else:
            # No feasible solution found
            sol_y = np.zeros(n_facilities)
            sol_x = np.zeros((n_facilities, n_facilities))
            obj_val = float('inf')
            bound = float('inf')

        # Build result dictionary
        result = {
            'problem_type': 'pmedian',
            'k': k,
            'transport_cost': instance.get("transport_cost"),
            'fixed_costs': instance.get("fixed_costs"),
            'nv': instance.get("nv"),
            'nc': instance.get("nc"),
            'seed': instance.get("seed"),
            'n_facilities': n_facilities,
            'mip_status': model.Status,
            'mip_obj_val': obj_val,
            'mip_runtime': model.Runtime,
            'mip_num_nodes': model.getAttr('NodeCount'),
            'mip_gap': model.MIPGap if model.SolCount > 0 else float('inf'),
            'best_bound': bound,
            'sol_facilities': sol_y,
            'sol_assignments': sol_x,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads
        }
        
        return result


# =============================================================================
# Solver Registry and Utility Functions
# =============================================================================

SOLVER_REGISTRY = {
    'uflp': UFLPSolver,
    'uncapacitated_facility_location': UFLPSolver,
    'pmedian': PMedianSolver,
    'p-median': PMedianSolver,
    'p_median': PMedianSolver
}


def get_solver(problem_type, **kwargs):
    """Get a solver instance for the specified problem type."""
    if problem_type not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown problem type: {problem_type}. "
                        f"Available types: {list(SOLVER_REGISTRY.keys())}")
    
    solver_class = SOLVER_REGISTRY[problem_type]
    return solver_class(**kwargs)


def solve_facility_location_problem(instance, problem_type=None, **solver_kwargs):
    """Convenience function to solve a facility location problem."""
    # Auto-detect problem type if not specified
    if problem_type is None:
        if 'k' in instance and instance.get('problem_type') in ['pmedian', 'p-median']:
            problem_type = 'pmedian'
        else:
            problem_type = 'uflp'
    
    solver = get_solver(problem_type, **solver_kwargs)
    return solver.solve(instance)


def validate_solution(instance, solution):
    """Validate a facility location solution."""
    try:
        n_facilities = solution['n_facilities']
        sol_facilities = solution['sol_facilities']
        sol_assignments = solution['sol_assignments']
        
        # Check dimensions
        if len(sol_facilities) != n_facilities:
            return False, f"Facility solution length {len(sol_facilities)} != {n_facilities}"
        
        if sol_assignments.shape != (n_facilities, n_facilities):
            return False, f"Assignment matrix shape {sol_assignments.shape} != ({n_facilities}, {n_facilities})"
        
        # Check that each client is assigned to exactly one facility
        assignment_sums = np.sum(sol_assignments, axis=1)
        if not np.allclose(assignment_sums, 1.0, atol=1e-6):
            return False, "Not all clients assigned to exactly one facility"
        
        # Check that clients are only assigned to open facilities
        for i in range(n_facilities):
            for j in range(n_facilities):
                if sol_assignments[i, j] > 1e-6 and sol_facilities[j] < 1e-6:
                    return False, f"Client {i} assigned to closed facility {j}"
        
        # Check k constraint for p-median
        if solution.get('problem_type') == 'pmedian':
            k = solution.get('k')
            open_facilities = np.sum(sol_facilities > 0.5)
            if abs(open_facilities - k) > 1e-6:
                return False, f"P-median: {open_facilities} facilities open, expected {k}"
        
        return True, "Valid solution"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def process_facility_location_file(instance_file, problem_type, solver_kwargs, results_folder=None):
    """Process a single facility location instance file: load, solve, and save results."""
    try:
        # Load instances
        with open(instance_file, 'rb') as f:
            instances = pickle.load(f)
        
        if not isinstance(instances, list):
            instances = [instances]
        
        # Get solver
        solver = get_solver(problem_type, **solver_kwargs)
        
        # Solve all instances
        results = []
        for i, instance in enumerate(instances):
            logging.info(f"Solving instance {i+1}/{len(instances)} from {instance_file}")
            result = solver.solve(instance)
            
            # Add metadata
            result['instance_file'] = instance_file
            result['instance_index'] = i
            
            # Validate solution
            is_valid, msg = validate_solution(instance, result)
            result['solution_valid'] = is_valid
            result['validation_message'] = msg
            
            results.append(result)
        
        # Save results
        base_name = os.path.basename(instance_file)
        prefix = base_name.replace(".pkl", "")
        
        if results_folder is None:
            results_folder = os.path.dirname(instance_file)
        
        result_file = os.path.join(results_folder, f"{prefix}_{problem_type}_results.pkl")
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        
        logging.info(f"Solved {len(results)} instances, saved to {result_file}")
        return results
        
    except Exception as e:
        logging.error(f"Error processing {instance_file}: {str(e)}")
        return {'error': str(e), 'instance_file': instance_file}


def main():
    """Command-line interface for facility location problem solving."""
    parser = argparse.ArgumentParser(description="Facility Location Problem Solver")
    parser.add_argument("--problem_type", type=str, required=True,
                       choices=list(SOLVER_REGISTRY.keys()),
                       help="Type of facility location problem to solve")
    parser.add_argument("--instance_files", type=str, nargs="+", required=True,
                       help="List of pickle files containing facility location instances")
    parser.add_argument("--time_limit", type=int, default=3600,
                       help="Time limit in seconds for each solve")
    parser.add_argument("--max_threads", type=int, default=1,
                       help="Maximum number of threads for Gurobi")
    parser.add_argument("--mip_gap", type=float, default=None,
                       help="MIP optimality gap tolerance")
    parser.add_argument("--mip_focus", type=int, default=None,
                       help="Gurobi MIP focus setting (0-3)")
    parser.add_argument("--nparallel", type=int, default=1,
                       help="Number of parallel solves")
    parser.add_argument("--results_folder", type=str, default=None,
                       help="Folder to save results (default: same as instance files)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Prepare solver kwargs
    solver_kwargs = {
        'time_limit': args.time_limit,
        'max_threads': args.max_threads,
        'mip_gap': args.mip_gap,
        'mip_focus': args.mip_focus
    }
    
    # Solve in parallel
    logging.info(f"Solving {args.problem_type} for {len(args.instance_files)} files")
    
    if HAS_TQDM_JOBLIB:
        with tqdm_joblib(tqdm(total=len(args.instance_files), 
                             desc=f"Solving {args.problem_type}", 
                             unit="file")):
            results = Parallel(n_jobs=args.nparallel)(
                delayed(process_facility_location_file)(
                    instance_file, args.problem_type, solver_kwargs, args.results_folder
                )
                for instance_file in args.instance_files
            )
    else:
        results = Parallel(n_jobs=args.nparallel)(
            delayed(process_facility_location_file)(
                instance_file, args.problem_type, solver_kwargs, args.results_folder
            )
            for instance_file in args.instance_files
        )
    
    # Summary statistics
    successful_results = [r for r in results if not isinstance(r, dict) or 'error' not in r]
    if successful_results:
        total_instances = sum(len(r) for r in successful_results)
        logging.info(f"Completed {len(successful_results)}/{len(results)} files")
        logging.info(f"Total instances solved: {total_instances}")
    else:
        logging.warning("No successful solves completed")


if __name__ == "__main__":
    main()
