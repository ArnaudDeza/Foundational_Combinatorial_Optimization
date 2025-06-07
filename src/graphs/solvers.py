#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import os
import glob
from pathlib import Path
from abc import ABC, abstractmethod
import pickle
import networkx as nx
 
import gurobipy as gp
from gurobipy import GRB
 
# For parallel solving
from joblib import Parallel, delayed
from tqdm import tqdm

from tqdm_joblib import tqdm_joblib
 

#from src import GRBENV
# =============================================================================
# Base and Solver Classes
# =============================================================================

class GraphProblemSolver(ABC):
    """Abstract base class for graph problem solvers."""

    def __init__(self, time_limit=None, max_threads=None, quadratic=False, prm_file=None):
        self.time_limit = time_limit
        self.max_threads = max_threads
        self.quadratic = quadratic
        self.prm_file = prm_file
 
    @abstractmethod
    def solve(self, graph):
        """
        Solve the given graph problem.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class MaxCutSolver(GraphProblemSolver):
    """Solver for the Max-Cut problem using a Gurobi formulation."""

    def solve(self, graph): 
        m = gp.Model("max_cut")
        m.setParam("OutputFlag", 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        # Create binary variables for each node.
        vdict = m.addVars(graph.number_of_nodes(), vtype=GRB.BINARY, name="x")

        # Build the objective function based on whether the graph is weighted.
        if nx.is_weighted(graph):
            cut_expr = [
                data['weight'] * (vdict[i] + vdict[j] - 2 * vdict[i] * vdict[j])
                for i, j, data in graph.edges(data=True)
            ]
        else:
            cut_expr = [
                (vdict[i] + vdict[j] - 2 * vdict[i] * vdict[j])
                for i, j in graph.edges()
            ]

        m.setObjective(gp.quicksum(cut_expr), GRB.MAXIMIZE)
        m.optimize()

        # Extract and return results.
        result = {
            'objective_value': m.ObjVal,
            'node_solution': [1 if abs(vdict[i].x) > 1e-6 else 0 for i in range(graph.number_of_nodes())],
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap,
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result


class MISolver(GraphProblemSolver):
    """Solver for the Maximum Independent Set (MIS) problem."""
    
    def solve(self, graph):
        if self.quadratic:
            return self.solve_quadratic(graph)
        else:
            return self.solve_linear(graph)
    
    def solve_quadratic(self, graph):
        """Solve MIS using quadratic formulation."""
        n = graph.number_of_nodes()
        adj = nx.to_numpy_array(graph)
        J = np.identity(n)
        A = J - adj

        m = gp.Model("mis_quad")
        m.setParam("OutputFlag", 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")
        m.setObjective(x @ A @ x, GRB.MAXIMIZE)

        if self.prm_file is not None:
            with open(self.prm_file, "r") as a_file:
                for line in a_file:
                    stripped_line = line.strip()
                    if stripped_line:
                        splitted = stripped_line.split()
                        if len(splitted) >= 2:
                            m.setParam(splitted[0], float(splitted[1]))
        else:
            if self.time_limit:
                m.setParam('ImproveStartTime', self.time_limit * 0.9)

        m.optimize()
 
        result = {
            'objective_value': m.ObjVal,
            'node_solution': [1 if abs(x[i].x) > 1e-6 else 0 for i in range(n)],
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap,
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result
    
    def solve_linear(self, graph):
        """Solve MIS using linear formulation."""
        m = gp.Model("mis_linear")
        m.setParam("OutputFlag", 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        # Create a binary variable for each node
        x = m.addVars(graph.nodes, vtype=GRB.BINARY, name="x")

        # Set the objective: maximize the number of selected nodes
        m.setObjective(gp.quicksum(x[i] for i in graph.nodes), GRB.MAXIMIZE)

        # For each edge, ensure that at most one of the two connected nodes is selected
        m.addConstrs((x[i] + x[j] <= 1 for i, j in graph.edges), name="edge_constraints")

        m.optimize()

        result = {
            'objective_value': m.objVal,
            'node_solution': [1 if x[i].x > 0.5 else 0 for i in graph.nodes],
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap,
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result


class MVCSolver(GraphProblemSolver):
    """Solver for the Minimum Vertex Cover (MVC) problem."""
    
    def solve(self, graph):
        m = gp.Model("mvc")
        m.setParam("OutputFlag", 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        # Create variable for each node
        x = m.addVars(graph.nodes, vtype=GRB.BINARY, name="x")

        # Objective function: minimize number of nodes
        m.setObjective(gp.quicksum(x[i] for i in graph.nodes), GRB.MINIMIZE)

        # Add constraint for each edge: at least one endpoint must be in cover
        m.addConstrs(x[i] + x[j] >= 1 for i, j in graph.edges)

        m.optimize()

        result = {
            'objective_value': m.objVal,
            'node_solution': [1 if x[i].x > 0.5 else 0 for i in graph.nodes],
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap,
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result


class MaxCliqueSolver(GraphProblemSolver):
    """Solver for the Maximum Clique problem."""
    
    def solve(self, graph):
        m = gp.Model("max_clique")
        m.setParam('OutputFlag', 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        # Create a binary variable for each node
        x = m.addVars(graph.nodes, vtype=GRB.BINARY, name="x")

        # Objective: maximize the number of nodes in the clique
        m.setObjective(gp.quicksum(x[i] for i in graph.nodes), GRB.MAXIMIZE)

        # Clique constraint: for every pair of nodes not connected by an edge,
        # at most one can be in the clique
        nodes = list(graph.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not graph.has_edge(nodes[i], nodes[j]):
                    m.addConstr(x[nodes[i]] + x[nodes[j]] <= 1)

        m.optimize()

        result = {
            'objective_value': m.objVal,
            'node_solution': [1 if x[i].x > 0.5 else 0 for i in graph.nodes],
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap,
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result


class GraphColoringSolver(GraphProblemSolver):
    """Solver for the Graph Coloring problem."""
    
    def __init__(self, max_colors=None, **kwargs):
        super().__init__(**kwargs)
        self.max_colors = max_colors
    
    def solve(self, graph):
        # Estimate maximum colors needed (conservative upper bound)
        if self.max_colors is None:
            max_colors = min(graph.number_of_nodes(), max(dict(graph.degree()).values()) + 1)
        else:
            max_colors = self.max_colors

        m = gp.Model("graph_coloring")
        m.setParam('OutputFlag', 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        # Binary variable: x[i,c] = 1 if node i gets color c
        x = m.addVars(graph.nodes, range(max_colors), vtype=GRB.BINARY, name="x")
        
        # Binary variable: y[c] = 1 if color c is used
        y = m.addVars(range(max_colors), vtype=GRB.BINARY, name="y")

        # Objective: minimize number of colors used
        m.setObjective(gp.quicksum(y[c] for c in range(max_colors)), GRB.MINIMIZE)

        # Each node gets exactly one color
        m.addConstrs(gp.quicksum(x[i, c] for c in range(max_colors)) == 1 
                    for i in graph.nodes)

        # Adjacent nodes cannot have the same color
        m.addConstrs(x[i, c] + x[j, c] <= 1 
                    for i, j in graph.edges 
                    for c in range(max_colors))

        # If a node uses color c, then color c must be used
        m.addConstrs(x[i, c] <= y[c] 
                    for i in graph.nodes 
                    for c in range(max_colors))

        m.optimize()

        # Check if solution is feasible
        if m.Status == GRB.OPTIMAL or (m.Status == GRB.TIME_LIMIT and m.SolCount > 0):
            try:
                # Extract node coloring
                node_colors = {}
                for i in graph.nodes:
                    for c in range(max_colors):
                        if x[i, c].X > 0.5:  # Use .X instead of .x for Gurobi variables
                            node_colors[i] = c
                            break
                
                obj_val = m.objVal
                num_colors = int(obj_val)
            except:
                # Fallback if variable access fails
                node_colors = {}
                obj_val = float('inf')
                num_colors = max_colors
        else:
            # No feasible solution found
            node_colors = {}
            obj_val = float('inf')
            num_colors = max_colors

        result = {
            'objective_value': obj_val,
            'node_colors': node_colors,
            'num_colors_used': num_colors,
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap if m.SolCount > 0 else float('inf'),
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result


class MinDominatingSetSolver(GraphProblemSolver):
    """Solver for the Minimum Dominating Set problem."""
    
    def solve(self, graph):
        m = gp.Model("min_dominating_set")
        m.setParam('OutputFlag', 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        # Create a binary variable for each node
        x = m.addVars(graph.nodes, vtype=GRB.BINARY, name="x")

        # Objective: minimize the size of the dominating set
        m.setObjective(gp.quicksum(x[i] for i in graph.nodes), GRB.MINIMIZE)

        # Dominating constraint: each node must be dominated
        # (either in the set or adjacent to a node in the set)
        for i in graph.nodes:
            neighbors = list(graph.neighbors(i))
            m.addConstr(x[i] + gp.quicksum(x[j] for j in neighbors) >= 1)

        m.optimize()

        result = {
            'objective_value': m.objVal,
            'node_solution': [1 if x[i].x > 0.5 else 0 for i in graph.nodes],
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap,
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result


class TotalDominatingSetSolver(GraphProblemSolver):
    """Solver for the Total Dominating Set problem."""
    
    def solve(self, graph):
        m = gp.Model("total_dominating_set")
        m.setParam('OutputFlag', 0)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        if self.max_threads is not None:
            m.setParam("Threads", self.max_threads)

        # Create a binary variable for each node
        x = m.addVars(graph.nodes, vtype=GRB.BINARY, name="x")

        # Objective: minimize the size of the total dominating set
        m.setObjective(gp.quicksum(x[i] for i in graph.nodes), GRB.MINIMIZE)

        # Total dominating constraint: each node must have at least one neighbor in the set
        for i in graph.nodes:
            neighbors = list(graph.neighbors(i))
            if neighbors:  # Only add constraint if node has neighbors
                m.addConstr(gp.quicksum(x[j] for j in neighbors) >= 1)

        m.optimize()

        result = {
            'objective_value': m.objVal,
            'node_solution': [1 if x[i].x > 0.5 else 0 for i in graph.nodes],
            'nodes_branched': m.getAttr('NodeCount'),
            'mipgap': m.MIPGap,
            'solver_runtime': m.Runtime,
            'time_limit': self.time_limit,
            'max_threads': self.max_threads,
            'status': m.Status
        }
        return result


# =============================================================================
# Solver Registry and Utility Functions
# =============================================================================

SOLVER_REGISTRY = {
    'max_cut': MaxCutSolver,
    'mis': MISolver,
    'maximum_independent_set': MISolver,
    'mvc': MVCSolver,
    'minimum_vertex_cover': MVCSolver,
    'max_clique': MaxCliqueSolver,
    'maximum_clique': MaxCliqueSolver,
    'graph_coloring': GraphColoringSolver,
    'coloring': GraphColoringSolver,
    'min_dominating_set': MinDominatingSetSolver,
    'dominating_set': MinDominatingSetSolver,
    'total_dominating_set': TotalDominatingSetSolver
}


def get_solver(problem_type, **kwargs):
    """Get a solver instance for the specified problem type."""
    if problem_type not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown problem type: {problem_type}. "
                        f"Available types: {list(SOLVER_REGISTRY.keys())}")
    
    solver_class = SOLVER_REGISTRY[problem_type]
    return solver_class(**kwargs)


def solve_graph_problem(graph, problem_type, **solver_kwargs):
    """Convenience function to solve a graph problem."""
    solver = get_solver(problem_type, **solver_kwargs)
    return solver.solve(graph)


def validate_solution(graph, problem_type, solution):
    """Validate a solution for a given graph problem."""
    node_solution = solution.get('node_solution', [])
    
    if problem_type in ['mis', 'maximum_independent_set']:
        # Check independence: no two selected nodes are adjacent
        selected_nodes = [i for i, val in enumerate(node_solution) if val == 1]
        for i in selected_nodes:
            for j in selected_nodes:
                if i != j and graph.has_edge(i, j):
                    return False, f"Nodes {i} and {j} are both selected but adjacent"
        return True, "Valid independent set"
    
    elif problem_type in ['mvc', 'minimum_vertex_cover']:
        # Check coverage: every edge has at least one endpoint in the cover
        selected_nodes = set(i for i, val in enumerate(node_solution) if val == 1)
        for i, j in graph.edges():
            if i not in selected_nodes and j not in selected_nodes:
                return False, f"Edge ({i}, {j}) not covered"
        return True, "Valid vertex cover"
    
    elif problem_type in ['max_clique', 'maximum_clique']:
        # Check clique: all selected nodes are pairwise adjacent
        selected_nodes = [i for i, val in enumerate(node_solution) if val == 1]
        for i in range(len(selected_nodes)):
            for j in range(i + 1, len(selected_nodes)):
                if not graph.has_edge(selected_nodes[i], selected_nodes[j]):
                    return False, f"Selected nodes {selected_nodes[i]} and {selected_nodes[j]} not adjacent"
        return True, "Valid clique"
    
    # Add more validation logic for other problem types as needed
    return True, f"Validation not implemented for {problem_type}"


def process_graph_file(graph_file, problem_type, solver_kwargs, results_folder=None):
    """Process a single graph file: load, solve, and save results."""
    try:
        # Load graph
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        
        # Solve problem
        solver = get_solver(problem_type, **solver_kwargs)
        result = solver.solve(graph)
        
        # Add metadata
        result['graph_file'] = graph_file
        result['problem_type'] = problem_type
        result['num_nodes'] = graph.number_of_nodes()
        result['num_edges'] = graph.number_of_edges()
        
        # Validate solution if possible
        is_valid, msg = validate_solution(graph, problem_type, result)
        result['solution_valid'] = is_valid
        result['validation_message'] = msg
        
        # Save results
        base_name = os.path.basename(graph_file)
        prefix = base_name.replace(".gpickle", "")
        
        if results_folder is None:
            results_folder = os.path.dirname(graph_file)
        
        out_file = os.path.join(results_folder, f"{prefix}_{problem_type}_results.npy")
        np.save(out_file, result)
        
        logging.info(f"Solved {problem_type} for {graph_file}, objective: {result.get('objective_value', 'N/A')}")
        return result
        
    except Exception as e:
        logging.error(f"Error processing {graph_file}: {str(e)}")
        return {'error': str(e), 'graph_file': graph_file}


def main():
    """Command-line interface for graph problem solving."""
    parser = argparse.ArgumentParser(description="Graph Combinatorial Optimization Solver")
    parser.add_argument("--problem_type", type=str, required=True,
                       choices=list(SOLVER_REGISTRY.keys()),
                       help="Type of graph problem to solve")
    parser.add_argument("--graph_files", type=str, nargs="+", required=True,
                       help="List of pickle files containing graph instances")
    parser.add_argument("--time_limit", type=int, default=3600,
                       help="Time limit in seconds for each solve")
    parser.add_argument("--max_threads", type=int, default=1,
                       help="Maximum number of threads for Gurobi")
    parser.add_argument("--nparallel", type=int, default=1,
                       help="Number of parallel solves")
    parser.add_argument("--results_folder", type=str, default=None,
                       help="Folder to save results (default: same as graph files)")
    parser.add_argument("--quadratic", action="store_true",
                       help="Use quadratic formulation where available")
    parser.add_argument("--max_colors", type=int, default=None,
                       help="Maximum number of colors for graph coloring")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Prepare solver kwargs
    solver_kwargs = {
        'time_limit': args.time_limit,
        'max_threads': args.max_threads,
        'quadratic': args.quadratic
    }
    
    if args.max_colors is not None:
        solver_kwargs['max_colors'] = args.max_colors
    
    # Solve in parallel
    logging.info(f"Solving {args.problem_type} for {len(args.graph_files)} graphs")
    
    with tqdm_joblib(tqdm(total=len(args.graph_files), desc=f"Solving {args.problem_type}", unit="graph")):
        results = Parallel(n_jobs=args.nparallel)(
            delayed(process_graph_file)(
                graph_file, args.problem_type, solver_kwargs, args.results_folder
            )
            for graph_file in args.graph_files
        )
    
    # Summary statistics
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        avg_objective = np.mean([r.get('objective_value', 0) for r in successful_results])
        avg_runtime = np.mean([r.get('solver_runtime', 0) for r in successful_results])
        logging.info(f"Completed {len(successful_results)}/{len(results)} instances")
        logging.info(f"Average objective value: {avg_objective:.2f}")
        logging.info(f"Average runtime: {avg_runtime:.2f}s")
    else:
        logging.warning("No successful solves completed")


if __name__ == "__main__":
    main()
 