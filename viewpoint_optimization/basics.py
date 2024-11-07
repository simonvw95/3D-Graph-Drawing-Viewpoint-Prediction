import torch
import numpy as np
import networkx as nx
import os
from pytorch3d.renderer import (look_at_view_transform)
from torch_geometric.data import Data

from viewpoint_optimization.metrics_gd_torch import *


"""
Creates an object containing the necessary data for all the optimization strategies

Input
G:          nx.Graph, a graph object
coords:     np.ndarray/torch.tensor, an array containing the 3D coordinates

Output
dataobj:    torch.Data, the torch data object containing the edge index, the 3d coordinates, the gtds in matrix and pair format and the 
            indices of the node pairs
"""


def get_data(G, coords, gtds_np = None):

    edges = np.array(G.edges())
    edge_idx = torch.tensor(np.array([np.concatenate((edges[:, 0], edges[:, 1])), np.concatenate((edges[:, 1], edges[:, 0]))]),
                            dtype=torch.int64)
    pos_3d = torch.tensor(coords).float()

    if gtds_np is None:
        gtds_np = nx.floyd_warshall_numpy(G)

    # instead of a large matrix we want to get the gtds into pairs (thereby excluding 0 values in the matrix and nans in gradient descent)
    tri_indices = np.triu_indices(np.shape(gtds_np)[0], 1)

    gtds_pairs = torch.tensor(gtds_np[tri_indices], dtype=torch.float32).unsqueeze(-1)
    idcs = torch.tensor(np.vstack((tri_indices[0], tri_indices[1])).T, dtype=torch.int)

    dataobj = Data(edge_index=edge_idx, x=pos_3d,  gtds=torch.tensor(gtds_np), gtds_pairs = gtds_pairs, idcs = idcs, degrees = G.degree(), name = G.name)

    return dataobj


def write_result(data_obj, loss):

    qm_name = data_obj.qm_function_name

    if not os.path.isdir('evaluations/results/' + qm_name):
        os.mkdir('evaluations/results/' + qm_name)

    full_name = qm_name + '-' + data_obj.name + '-' + data_obj.strategy + '-' + str(data_obj.N_max)
    if hasattr(data_obj, 'iteration'):
        full_name += '-i' + str(data_obj.iteration)
    f = open('evaluations/results/' + qm_name + '/' + full_name + '.txt', 'a')
    f.write(str(1 - loss.item()) + '\n')
    f.close()


"""
Computes the loss of a given graph, 3d coordinates and viewing angles

Input
optim_param:        tuple, a tuple of the elevation and azimuth
*args:              tuple, a tuple of the torch data object, a dictionary holding quality metrics and metric weights, the (perspective)
                    camera object (PerspectiveCameras) from pytorch3d, a boolean indicating whether the result is written to a txt file or not
Output
tot_loss:           torch.tensor, the loss/quality metric value

in case of the newton_raphson being used the output changes to
tot_loss:           torch.tensor.item()/float, the loss/quality metric value
grad:               the gradient of the loss function
"""


def loss_function(optim_param, *args):

    # extract the parameters
    elevation, azimuth = optim_param
    data_obj, qm_function, camera, write = args

    # turn into tensors and float in case they are not
    if type(elevation) != torch.Tensor:
        elevation = torch.tensor(elevation)
    if type(azimuth) != torch.Tensor:
        azimuth = torch.tensor(azimuth)

    elevation = elevation.float()
    azimuth = azimuth.float()

    # for newton raphson we want to return the gradient so some changes need to happen if we're doing newt-raph
    if 'newton_raphson' in data_obj.strategy:
        newt_raphs = True
    else:
        newt_raphs = False

    if newt_raphs:
        elevation.requires_grad_(True)
        azimuth.requires_grad_(True)

    # get the transformation matrix
    R, T = look_at_view_transform(1, elevation, azimuth, camera.device)
    transform_matrix = camera.get_full_projection_transform(R=R.float().to(camera.device), T=T.float().to(camera.device)).get_matrix()[0]

    # apply the transformation matrix to the 3d coords to get 2d coords (view)
    coords_3d = data_obj.x
    n = coords_3d.shape[0]
    projection = torch.ones((n, 4))

    projection[0:n, :3] = coords_3d
    view = torch.matmul(projection.to(camera.device), transform_matrix.float())[:, 0:2]

    # Scale exactly to the range (0, 1)
    view = view - torch.min(view, axis=0)[0]
    coords_2d = view / torch.max(view)

    # the qm_function variable is a dictionary containing the function and its weight
    tot_loss = 0
    qm_funcs = list(qm_function.keys())

    write_loss = 0
    n_qms = 0
    # loop over all quality metrics
    for qm_func in qm_funcs:
        # only compute it if it actually has a weight (Nones are excluded this way)
        if qm_function[qm_func]:
            res = qm_func(coords_2d, data_obj)
            n_qms += 1
            # handle the case of the crossing number approximation which returns 3 values (the approx loss, the approx cn, and the actual cn)
            if isinstance(res, tuple):
                loss = res[0]
                write_loss += res[2]
            else:
                loss = res
                write_loss += res
            # add to the total loss, weighted by the value in the dict
            tot_loss = tot_loss + qm_function[qm_func] * loss
    tot_loss = tot_loss / n_qms

    # write out the result to the txt file
    if write:
        write_result(data_obj, write_loss)

    # return the gradient and perform backpropagation here if newton-raphson method
    if not newt_raphs:
        return tot_loss
    else:
        # perform backpropagation to compute the gradients
        tot_loss.backward()

        # extract the gradients for elevation and azimuth
        grad = torch.tensor([elevation.grad.item(), azimuth.grad.item()])

        # return the loss value and gradient
        return tot_loss.item(), grad.numpy()


# slight adjustment of the previous one, this one loops over if multiple angles are given (needed as a single function for swarm operations)
def loss_function_mult(optim_param_mult, data_obj, qm_function, camera, write):

    losses = []
    for angles in optim_param_mult:
        optim_param = angles
        loss = loss_function(optim_param, data_obj, qm_function, camera, write)
        losses.append(loss)

    return losses
