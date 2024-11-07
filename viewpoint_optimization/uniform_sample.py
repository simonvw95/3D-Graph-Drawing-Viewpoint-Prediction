import numpy as np

from utils import fibonacci_sphere
from utils import rectangular_to_spherical, spherical_to_rectangular
from viewpoint_optimization.basics import loss_function


def eval_points(points, data_obj, qm_function, camera, write):

    spherical_coords = rectangular_to_spherical(points)

    losses = [0] * len(spherical_coords)
    for i in range(len(spherical_coords)):
        loss = loss_function((spherical_coords[i][0], spherical_coords[i][1]), data_obj, qm_function, camera, write)
        losses[i] = loss

    best_loss, best_angle_idx = np.min(losses), np.argmin(losses)
    worst_loss, worst_angle_idx = np.max(losses), np.argmax(losses)

    uni_results = {'init_points' : points, 'best_val' : best_loss, 'best_angle' : points[best_angle_idx], 'best_angle_idx' : best_angle_idx, 'worst_val' : worst_loss, 'worst_angle' : points[worst_angle_idx]}

    return uni_results


def run_uni_sample(N, data_obj, qm_function, camera, bounds, write = False):

    # get the initial uniform covering sample
    # since we only want to consider N points on the half sphere, then we simply multiply N by 2 so that the half sphere has N points
    points = fibonacci_sphere(N * 2)

    spherical_coords = rectangular_to_spherical(points)
    # filter out half of the sphere based on bounds
    # filter based on elevation, then azimuth
    bounds = np.array(bounds)
    spherical_coords = spherical_coords[(spherical_coords[:, 0] >= bounds[0][0]) & (spherical_coords[:, 0] < bounds[0][1])]
    spherical_coords = spherical_coords[(spherical_coords[:, 1] >= bounds[1][0]) & (spherical_coords[:, 1] < bounds[1][1])]
    points = spherical_to_rectangular(spherical_coords)

    uni_results = eval_points(points, data_obj, qm_function, camera, write)

    return uni_results
