from scipy.optimize import differential_evolution
from viewpoint_optimization.basics import *


def run_genetic_algorithm(N, popsize, bounds, data_obj, qm_function, camera, write):

    # max iterations is the maximum number of iterations but not the maximum number of evaluations of the object function
    # due to the popolation size that the differential evolution takes
    # number of total evaluations: (max_iter + 1) * pop_size * P
    # where P is the number of unique bounds which in our case is 2
    params = (data_obj, qm_function, camera, write)

    results = differential_evolution(func = loss_function, bounds = bounds, args = params, maxiter = N, popsize = popsize, tol = 0.00000001)

    return results