import torch.nn as nn
import numpy as np
import time
import copy

from pytorch3d.renderer import (PerspectiveCameras)
from viewpoint_optimization.basics import loss_function
from viewpoint_optimization.metrics_gd_torch import *


# class for the model that will optimize the viewing angles
class Model(nn.Module):
    def __init__(self, data, device, start_viewpoint, qm_function, write, bounds):
        super().__init__()
        self.data = data
        self.device = device
        self.camera = PerspectiveCameras(device = device, focal_length = torch.tensor([1]).float().to(device))
        self.start_viewpoint = start_viewpoint
        self.qm_function = qm_function
        self.camera_position = nn.Parameter(torch.from_numpy(self.start_viewpoint).to(self.device))
        self.bounds = torch.tensor(bounds).to(self.device)
        self.write = write
        # self.elevation_min = self.bounds[0][0]
        # self.elevation_max = self.bounds[0][1]
        # self.azimuth_min = self.bounds[1][0]
        # self.azimuth_max = self.bounds[1][1]

    def forward(self):
        # restrict the predicted viewpoint to a certain range (default is half sphere)
        # self.camera_position.data[0] = torch.clamp(self.camera_position.data[0], self.elevation_min, self.elevation_max)
        # self.camera_position.data[1] = torch.clamp(self.camera_position.data[1], self.azimuth_min, self.azimuth_max)

        loss = loss_function((self.camera_position[0].float(), self.camera_position[1].float()), self.data, self.qm_function, self.camera, self.write)

        return loss


# function for optimizing the viewpoint given starting viewpoint, the data, and a quality metric function
def optimize_viewpoint(start_viewpoint, curr_graph, qm_function, device, verbose, N, lr, bounds, write = False):

    # initialize the model and copy the starting viewpoint
    model = Model(data=curr_graph, device=device, start_viewpoint=copy.deepcopy(start_viewpoint), qm_function = qm_function, bounds = bounds, write = write).to(device)
    viewpoint_progression = [copy.deepcopy(start_viewpoint)]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # simple loop for gradient descent optimization
    loop = range(N)

    losses = []
    for i in loop:
        optimizer.zero_grad()
        # print(model.camera_position)
        loss_i = model()
        # print(loss_i)
        losses.append(1 - loss_i)

        if verbose:
            if (i % 100) == 0:
                print('on iteration: ' + str(i) + ' loss was: ' + str(1 - loss_i.item()))

        if (i % 10 == 0) and (i > 0):
            viewpoint_progression.append(copy.deepcopy(model.start_viewpoint))
        loss_i.backward()
        optimizer.step()

    results = {tuple(model.start_viewpoint) : loss_i.item()}, viewpoint_progression

    return results


# function for taking a graph and 3D coords as input and the starting viewpoint, and spits out the best result
def run_gradient_descent(curr_viewpoint, data_obj, N = 1000, lr = 0.15, device = 'cpu', verbose = False, qm_function = norm_stress_torch_pairs, bounds = list(zip(torch.tensor([-90, 90]), torch.tensor([-90, 90]))), write = False):

    curr_graph = data_obj
    # start_normal = time.time()
    results, vwp_progression = optimize_viewpoint(start_viewpoint= curr_viewpoint, curr_graph = curr_graph, qm_function = qm_function, verbose = verbose, device = device, N = N, lr = lr, bounds = bounds, write = write)
    # end_normal = time.time()
    # print('time taken: ' + str(round(end_normal - start_normal, 2)))

    best_vwp = max(results, key=results.get)

    return {'best_vwp': np.append(best_vwp, 1), 'best_val': results[best_vwp], 'viewpoint_progression' : vwp_progression}
