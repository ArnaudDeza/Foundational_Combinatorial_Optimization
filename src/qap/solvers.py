from gurobipy import Model, GRB, quicksum 
from scipy.spatial import distance_matrix
import time
import torch

def calculate_distances(positions):
    """
    Calculate a all distances between poistions

    :param np.array positions: Positions of (tour_len, 2) points
    :return: list with all distances
    """ 
    distances = distance_matrix(positions, positions)
    return distances

def solve_one_qap(instance: dict, time_limit: int) -> dict:
    """
    Solve a single QAP instance using Gurobi and return a dictionary with results.
    
    The instance dictionary should contain:
        - "F": Flow matrix (numpy.ndarray)
        - "positions": Node coordinates (numpy.ndarray)
    
    The distance matrix is computed using utils.distance_matrix.
    
    Args:
        instance (dict): QAP instance data.
        time_limit (int): Time limit (in seconds) for the Gurobi solver.
    
    Returns:
        dict: A dictionary with keys:
            "F", "positions", "D", "objVal", "mipgap", "runtime", "solution", "time_limit"
    """
    t0 = time.time()
    F = instance["F"]
    positions = instance["positions"]
    N = F.shape[0]

    # Ensure positions are in torch.Tensor format for distance computation.
    if not isinstance(positions, torch.Tensor):
        positions_tensor = torch.from_numpy(positions)
    else:
        positions_tensor = positions

    # Compute the distance matrix (assumed to be symmetric).
    D = calculate_distances(positions_tensor) 

    # Set up the QAP model.
    model = Model("QAP")
    model.Params.OutputFlag = 0  # Suppress output.
    model.Params.TimeLimit = time_limit

    # Decision variable: x[i,k] = 1 if facility i is assigned to location k.
    x = model.addMVar(shape=(N, N), vtype=GRB.BINARY, name="x")

    # Objective: Minimize sum_{i,j,k,l} F[i,j] * D[k,l] * x[i,k] * x[j,l]
    model.setObjective(quicksum(quicksum(F * (x @ D @ x.T))), GRB.MINIMIZE)

    # Assignment constraints: each facility is assigned to exactly one location.
    for i in range(N):
        model.addConstr(quicksum(x[i, j] for j in range(N)) == 1, name=f"row_{i}")
    for j in range(N):
        model.addConstr(quicksum(x[i, j] for i in range(N)) == 1, name=f"col_{j}")

    # Solve the model.
    model.optimize()
    elapsed = time.time() - t0

    # If an optimal solution is found (or the time limit is hit), extract the solution.
    solution = None
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        solution = x.X.copy()

    result = {
        "F": F,
        "positions": positions,
        "D": D,
        "objVal": model.objVal if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None,
        "mipgap": model.MIPGap if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None,
        "runtime": model.Runtime if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else elapsed,
        "solution": solution,
        "time_limit": time_limit,
    }
    return result

