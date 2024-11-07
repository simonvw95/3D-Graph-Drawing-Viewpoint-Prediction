from scipy.optimize import fmin_tnc
from viewpoint_optimization.basics import *


def run_newton_raphson(N, bounds, data_obj, qm_function, camera, curr_viewpoint, write = False):

    # do note that maxiter is the maximum number of iterations (you can have more than one call to the objective function (qm function) in an iteration
    # so maxfun is the maximum number of calls to the loss function, which can sometimes be exceeded
    params = (data_obj, qm_function, camera, write)
    results = fmin_tnc(func = loss_function, bounds = bounds, args = params, maxfun = N, x0 = curr_viewpoint, disp = 0, ftol = 0.0, pgtol = 0.0, xtol = 0.0, accuracy = 0.0)
    best_loss = loss_function(results[0], data_obj, qm_function, camera, write)

    return {'best_vwp': np.append(results[0], 1), 'best_val': best_loss[0]}
