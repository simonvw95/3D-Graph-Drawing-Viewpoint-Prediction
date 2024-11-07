import copy

from viewpoint_optimization.basics import *
from viewpoint_optimization.uniform_sample import run_uni_sample, eval_points
from utils import rectangular_to_spherical, spherical_to_rectangular


def generate_grid_sample(N, all_prev_points, selected_point):

    # get euclidean distance to all previous points
    distances = np.linalg.norm(all_prev_points - selected_point, axis=1)

    # get the closest point
    closest_point = all_prev_points[np.argmin(distances[np.nonzero(distances)])]

    # turn them to spherical angles
    selected_point_sph = rectangular_to_spherical(np.expand_dims(selected_point, axis = 0))
    closest_point_sph = rectangular_to_spherical(np.expand_dims(closest_point, axis=0))

    center_theta_deg = selected_point_sph[0][0]
    cap_theta_deg = closest_point_sph[0][0]

    center_phi_deg = selected_point_sph[0][1]
    cap_phi_deg = closest_point_sph[0][1]

    # determine the min and max degrees for the new sampling area
    diff_theta = abs(center_theta_deg - cap_theta_deg)
    diff_phi = abs(center_phi_deg - cap_phi_deg)

    if diff_theta >= diff_phi:
        diff = diff_theta
    else:
        diff = diff_phi

    theta_min_deg = center_theta_deg - diff
    theta_max_deg = center_theta_deg + diff
    phi_min_deg = center_phi_deg - diff
    phi_max_deg = center_phi_deg + diff

    # if we want to sample N points then we need sqrt of N points along one axis and the same along the other to make a grid
    n_points_single_axis = int(np.sqrt(N))

    # create a sampling grid
    theta_grid_deg = np.linspace(theta_min_deg, theta_max_deg, n_points_single_axis)  # Elevation angles in degrees
    phi_grid_deg = np.linspace(phi_min_deg, phi_max_deg, n_points_single_axis)  # Azimuth angles in degrees

    theta_grid, phi_grid = np.meshgrid(theta_grid_deg, phi_grid_deg)

    # turn into cartesian coordinates
    grid = np.vstack((theta_grid.flatten(), phi_grid.flatten())).T

    grid = np.hstack((grid, np.ones((len(grid), 1))))

    grid = spherical_to_rectangular(grid)

    return grid


def run_iter_resampling(N, divisions, data_obj, qm_function, camera, bounds, write = False):

    # get a uniform covering sampling using the run_uni_sample function
    uni_results = run_uni_sample(N, data_obj, qm_function, camera, bounds, write)
    # get the best viewpoint results
    best_angle = uni_results['best_angle']
    best_angle_idx = uni_results['best_angle_idx']
    prev_points = copy.deepcopy(uni_results['init_points'])
    # keep track of these best viewpoints for later visualization if necessary
    best_curr_val = uni_results['best_val']
    best_angles = [best_angle]
    all_prev_points = prev_points
    worst_losses = [uni_results['worst_val']]
    selected_point = prev_points[best_angle_idx]
    # loop over the number of divisions
    for d in range(divisions):

        # get sample points on the spherical cap surrounding the best viewpoint and evaluate those viewpoints
        # cap_points, cap_sample_points = sample_spherical_cap(N, prev_points, best_angle_idx)

        # generate new sample points in a grid around the best viewpoint
        new_sample_grid = generate_grid_sample(N, all_prev_points, selected_point)
        # evaluate all new sampled points
        layer_results = eval_points(new_sample_grid, data_obj, qm_function, camera, write)
        # if any of those sampled points are better then replace the center and the best found value
        if layer_results['best_val'] < best_curr_val:
            selected_point = layer_results['best_angle']
            best_curr_val = layer_results['best_val']

        # add the sampled points to a list of all points, from this we then construct a new grid
        prev_points = copy.deepcopy(layer_results['init_points'])
        all_prev_points = np.vstack((all_prev_points, prev_points))

        best_angles.append(copy.deepcopy(layer_results['best_angle']))
        worst_losses.append(layer_results['worst_val'])

    if divisions < 1:
        final_result = uni_results
    else:
        final_result = layer_results

    final_result['worst_val'] = np.max(worst_losses)

    return final_result
