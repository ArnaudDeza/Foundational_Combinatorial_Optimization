#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import os
import glob
import pickle
import networkx as nx
from pathlib import Path
import json

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

# Import solvers from the same directory
try:
    from .solvers import get_solver, SOLVER_REGISTRY, validate_solution
except ImportError:
    # Fallback for when running as script
    from solvers import get_solver, SOLVER_REGISTRY, validate_solution


def load_graph_instance(graph_file):
    """Load a graph instance from a gpickle file."""
    try:
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except Exception as e:
        logging.error(f"Error loading graph from {graph_file}: {str(e)}")
        return None


def solve_single_instance(graph_file, problem_type, solver_kwargs, results_folder=None, 
                         save_individual=True, overwrite=False):
    """Solve a single graph instance and optionally save results."""
    try:
        # Check if results already exist
        base_name = os.path.basename(graph_file)
        prefix = base_name.replace(".gpickle", "")
        
        if results_folder is None:
            results_folder = os.path.dirname(graph_file)
        
        result_file = os.path.join(results_folder, f"{prefix}_{problem_type}_results.npy")
        
        if not overwrite and os.path.exists(result_file):
            logging.info(f"Results already exist for {graph_file}, skipping")
            # Load and return existing results
            try:
                return np.load(result_file, allow_pickle=True).item()
            except:
                logging.warning(f"Could not load existing results from {result_file}, recomputing")
        
        # Load graph
        graph = load_graph_instance(graph_file)
        if graph is None:
            return {'error': 'Failed to load graph', 'graph_file': graph_file}
        
        # Get solver and solve
        solver = get_solver(problem_type, **solver_kwargs)
        result = solver.solve(graph)
        
        # Add metadata
        result['graph_file'] = graph_file
        result['problem_type'] = problem_type
        result['num_nodes'] = graph.number_of_nodes()
        result['num_edges'] = graph.number_of_edges()
        result['graph_density'] = nx.density(graph)
        
        # Validate solution if possible
        is_valid, msg = validate_solution(graph, problem_type, result)
        result['solution_valid'] = is_valid
        result['validation_message'] = msg
        
        # Save individual results if requested
        if save_individual:
            os.makedirs(results_folder, exist_ok=True)
            np.save(result_file, result)
            logging.debug(f"Saved results for {graph_file} to {result_file}")
        
        # Log progress
        obj_val = result.get('objective_value', 'N/A')
        runtime = result.get('solver_runtime', 0)
        logging.info(f"Solved {problem_type} for {prefix}: obj={obj_val}, time={runtime:.2f}s")
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing {graph_file}: {str(e)}")
        return {'error': str(e), 'graph_file': graph_file, 'problem_type': problem_type}


def solve_multiple_instances(graph_files, problem_type, solver_kwargs, 
                           results_folder=None, nparallel=1, save_individual=True, 
                           overwrite=False, progress_bar=True):
    """Solve multiple graph instances in parallel."""
    
    logging.info(f"Solving {problem_type} for {len(graph_files)} graph instances")
    logging.info(f"Using {nparallel} parallel workers")
    
    # Setup results folder
    if results_folder is not None:
        os.makedirs(results_folder, exist_ok=True)
    
    # Solve in parallel with progress bar
    if progress_bar:
        with tqdm_joblib(tqdm(total=len(graph_files), 
                             desc=f"Solving {problem_type}", 
                             unit="graph")):
            results = Parallel(n_jobs=nparallel)(
                delayed(solve_single_instance)(
                    graph_file, problem_type, solver_kwargs, 
                    results_folder, save_individual, overwrite
                )
                for graph_file in graph_files
            )
    else:
        results = Parallel(n_jobs=nparallel)(
            delayed(solve_single_instance)(
                graph_file, problem_type, solver_kwargs, 
                results_folder, save_individual, overwrite
            )
            for graph_file in graph_files
        )
    
    return results


def analyze_results(results):
    """Analyze and summarize solving results."""
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    summary = {
        'total_instances': len(results),
        'successful_solves': len(successful_results),
        'failed_solves': len(failed_results),
        'success_rate': len(successful_results) / len(results) if results else 0.0
    }
    
    if successful_results:
        objectives = [r.get('objective_value', 0) for r in successful_results]
        runtimes = [r.get('solver_runtime', 0) for r in successful_results]
        num_nodes = [r.get('num_nodes', 0) for r in successful_results]
        num_edges = [r.get('num_edges', 0) for r in successful_results]
        
        summary.update({
            'avg_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'min_objective': np.min(objectives),
            'max_objective': np.max(objectives),
            'avg_runtime': np.mean(runtimes),
            'std_runtime': np.std(runtimes),
            'min_runtime': np.min(runtimes),
            'max_runtime': np.max(runtimes),
            'avg_nodes': np.mean(num_nodes),
            'avg_edges': np.mean(num_edges),
            'total_runtime': np.sum(runtimes)
        })
        
        # Validation summary
        valid_solutions = [r for r in successful_results if r.get('solution_valid', False)]
        summary['valid_solutions'] = len(valid_solutions)
        summary['validation_rate'] = len(valid_solutions) / len(successful_results)
    
    return summary


def save_summary_results(results, summary, output_file):
    """Save comprehensive results and summary to files."""
    # Save detailed results
    results_file = output_file.replace('.json', '_detailed.npy')
    np.save(results_file, results)
    
    # Save summary as JSON
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Saved detailed results to {results_file}")
    logging.info(f"Saved summary to {output_file}")


def find_graph_files(root_directories, pattern="*.gpickle"):
    """Find all graph files matching the pattern in given directories."""
    all_files = []
    
    for root_dir in root_directories:
        if os.path.isfile(root_dir):
            # Single file provided
            all_files.append(root_dir)
        elif os.path.isdir(root_dir):
            # Directory provided, search recursively
            pattern_path = os.path.join(root_dir, "**", pattern)
            files = glob.glob(pattern_path, recursive=True)
            all_files.extend(files)
        else:
            logging.warning(f"Path does not exist: {root_dir}")
    
    return sorted(all_files)


def main():
    """Main entry point for the graph instance solver."""
    parser = argparse.ArgumentParser(
        description="Solve multiple graph combinatorial optimization instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument("--graph_files", type=str, nargs="+", 
                       help="Graph files or directories containing .gpickle files")
    parser.add_argument("--problem_type", type=str, required=True,
                       choices=list(SOLVER_REGISTRY.keys()),
                       help="Type of graph problem to solve")
    
    # Solver parameters
    parser.add_argument("--time_limit", type=int, default=3600,
                       help="Time limit in seconds for each solve")
    parser.add_argument("--max_threads", type=int, default=1,
                       help="Maximum threads for Gurobi solver")
    parser.add_argument("--quadratic", action="store_true",
                       help="Use quadratic formulation where available (e.g., MIS)")
    parser.add_argument("--max_colors", type=int, default=None,
                       help="Maximum colors for graph coloring problem")
    
    # Parallel processing
    parser.add_argument("--nparallel", type=int, default=1,
                       help="Number of parallel solver processes")
    
    # Output control
    parser.add_argument("--results_folder", type=str, default=None,
                       help="Folder to save results (default: same as graph files)")
    parser.add_argument("--output_summary", type=str, default=None,
                       help="File to save summary results (JSON format)")
    parser.add_argument("--save_individual", action="store_true", default=True,
                       help="Save individual .npy result files")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing result files")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--no_progress", action="store_true",
                       help="Disable progress bar")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Find graph files
    if args.graph_files:
        graph_files = find_graph_files(args.graph_files)
    else:
        logging.error("No graph files or directories specified")
        return 1
    
    if not graph_files:
        logging.error("No .gpickle files found in specified locations")
        return 1
    
    logging.info(f"Found {len(graph_files)} graph files")
    
    # Prepare solver kwargs
    solver_kwargs = {
        'time_limit': args.time_limit,
        'max_threads': args.max_threads,
        'quadratic': args.quadratic
    }
    
    if args.max_colors is not None:
        solver_kwargs['max_colors'] = args.max_colors
    
    # Solve instances
    results = solve_multiple_instances(
        graph_files=graph_files,
        problem_type=args.problem_type,
        solver_kwargs=solver_kwargs,
        results_folder=args.results_folder,
        nparallel=args.nparallel,
        save_individual=args.save_individual,
        overwrite=args.overwrite,
        progress_bar=not args.no_progress
    )
    
    # Analyze results
    summary = analyze_results(results)
    
    # Print summary
    print(f"\n{'='*50}")
    print("SOLVING SUMMARY")
    print(f"{'='*50}")
    print(f"Problem type: {args.problem_type}")
    print(f"Total instances: {summary['total_instances']}")
    print(f"Successful solves: {summary['successful_solves']}")
    print(f"Failed solves: {summary['failed_solves']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    if summary['successful_solves'] > 0:
        print(f"\nObjective Statistics:")
        print(f"  Average: {summary['avg_objective']:.2f}")
        print(f"  Std Dev: {summary['std_objective']:.2f}")
        print(f"  Min: {summary['min_objective']:.2f}")
        print(f"  Max: {summary['max_objective']:.2f}")
        
        print(f"\nRuntime Statistics:")
        print(f"  Average: {summary['avg_runtime']:.2f}s")
        print(f"  Std Dev: {summary['std_runtime']:.2f}s")
        print(f"  Min: {summary['min_runtime']:.2f}s")
        print(f"  Max: {summary['max_runtime']:.2f}s")
        print(f"  Total: {summary['total_runtime']:.2f}s")
        
        print(f"\nGraph Statistics:")
        print(f"  Average nodes: {summary['avg_nodes']:.1f}")
        print(f"  Average edges: {summary['avg_edges']:.1f}")
        
        print(f"\nValidation:")
        print(f"  Valid solutions: {summary['valid_solutions']}")
        print(f"  Validation rate: {summary['validation_rate']:.1%}")
    
    # Save summary if requested
    if args.output_summary:
        save_summary_results(results, summary, args.output_summary)
    
    return 0 if summary['success_rate'] > 0 else 1


if __name__ == "__main__":
    exit(main())

