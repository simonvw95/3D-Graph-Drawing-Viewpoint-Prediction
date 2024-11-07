from scipy.optimize import dual_annealing
from viewpoint_optimization.basics import *


def run_sim_annealing(N, bounds, data_obj, qm_function, camera, curr_viewpoint, temp, write = False):

    # do note that maxiter is the maximum number of iterations (you can have more than one call to the objective function (qm function) in an iteration
    # so maxfun is the maximum number of calls to the loss function, which can sometimes be exceeded
    params = (data_obj, qm_function, camera, write)
    results = dual_annealing(func = loss_function, bounds = bounds, args = params, maxiter = N, maxfun = N, x0 = curr_viewpoint, initial_temp = temp)

    return results
