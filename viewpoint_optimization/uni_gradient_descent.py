from viewpoint_optimization.gradient_descent import *
from utils import rectangular_to_spherical
from utils import fibonacci_sphere


def init_file(data_obj):

    full_name = data_obj.qm_function_name + '-' + data_obj.name + '-' + data_obj.strategy + '-' + str(data_obj.N_max)
    full_name += '-i' + str(data_obj.iteration)

    with open('evaluations/results/' + data_obj.qm_function_name + '/' + full_name + '.txt', "a") as file:
        file.write("start\n")
        file.close()


def run_uni_grad_descent(N_start, N_grad, lr, data_obj, qm_function, bounds, write = False):

    # since we only want to consider N points on the half sphere, then we simply multiply N by 2 so that the half sphere has N points
    points = fibonacci_sphere(N_start * 2)

    spherical_coords = rectangular_to_spherical(points)
    # filter out half of the sphere based on bounds
    # filter based on elevation, then azimuth
    bounds = np.array(bounds)
    spherical_coords = spherical_coords[
        (spherical_coords[:, 0] >= bounds[0][0]) & (spherical_coords[:, 0] < bounds[0][1])]
    angles = spherical_coords[
        (spherical_coords[:, 1] >= bounds[1][0]) & (spherical_coords[:, 1] < bounds[1][1])]
    angles = angles[:, :2]

    results = {}

    # do gradient descent to find the best viewpoint out of multiple starting viewpoints
    best_loss = 1
    best_vwp = angles[0]
    for angl in angles:
        if write:
            init_file(data_obj)
        curr_results = run_gradient_descent(curr_viewpoint = angl, data_obj = data_obj, N = N_grad, lr = lr, qm_function = qm_function, bounds = bounds, write = write)
        results[tuple(angl)] = curr_results

        if curr_results['best_val'] < best_loss:
            best_vwp = angl
            best_loss = curr_results['best_val']

    return {'best_vwp' : np.append(best_vwp, 1), 'best_val' : best_loss}
