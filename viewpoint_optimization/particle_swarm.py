from pyswarms.single.global_best import GlobalBestPSO

from viewpoint_optimization.basics import *


def run_pso(N, particles, l_bound, u_bound, data_obj, qm_function, camera, write = False):

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=particles, dimensions=2, options=options, bounds=(l_bound, u_bound))

    cost, pos = optimizer.optimize(loss_function_mult, N, data_obj = data_obj, qm_function = qm_function, camera = camera, write = write, verbose = False)

    return cost, pos
