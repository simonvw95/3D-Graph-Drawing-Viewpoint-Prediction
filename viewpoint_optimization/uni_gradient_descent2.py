from viewpoint_optimization.gradient_descent import *
from utils import rectangular_to_spherical
from utils import fibonacci_sphere


def run_uni_grad_descent_v2(N_start, N_grad, lr, data_obj, qm_function_norm, qm_function_grad, bounds, camera, write = False):

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

    losses = []
    for angle in angles:
        losses.append(loss_function(angle, data_obj, qm_function_norm, camera, write))

    best_angle = angles[torch.argmin(torch.tensor(losses)).item()]

    # do gradient descent to find the best viewpoint out of multiple starting viewpoints
    curr_results = run_gradient_descent(curr_viewpoint=best_angle, data_obj=data_obj, N=N_grad, lr=lr, qm_function=qm_function_grad, bounds=bounds, write = write)
    best_loss = curr_results['best_val']

    return {'best_vwp' : curr_results['best_vwp'], 'best_val' : best_loss}
