'''
Exact solvers for QUBO problems
'''
import time
from gurobi_optimods.qubo import solve_qubo
import gurobipy as gp
from gurobipy import GRB

def solve_one_qubo__GUROBI_OPTIMODS(Q, time_limit):
    """
    Solve a single QUBO matrix and return a dictionary with results.
    """
    t0 = time.time()
    res = solve_qubo(Q, time_limit=time_limit)
    elapsed = time.time() - t0
    return {
        "Q": Q,
        "objective": res.objective_value,
        "solution": res.solution,
        "runtime": elapsed,
        "time_limit":time_limit
    }

def solve_one_qubo__GUROBI(Q, time_limit, max_threads):
    """
    Solve a single QUBO matrix and return a dictionary with results.
    """
    n = Q.shape[0]
    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads   = max_threads

    x = model.addMVar(shape=n, vtype=GRB.BINARY, name="x")
    model.setObjective(x @ Q @ x, GRB.MINIMIZE)
    model.optimize()

    status   = model.Status
    runtime  = model.Runtime
    mipgap   = model.MIPGap if hasattr(model, "MIPGap") else None
    objective = model.ObjVal if model.SolCount > 0 else None

    solution = []
    if model.SolCount > 0:
        for i in range(n):
            solution.append(int(x[i].X))

    return {
        "Q": Q,
        "solution": solution,
        "runtime": runtime,
        "status": status,
        "mipgap": mipgap,
        "objective": objective,
        "time_limit":time_limit,
        "max_threads":max_threads
    }