from ...functions import *
import torch
from torch import nn
import torch_scatter
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras)
from pytorch3d.transforms.transform3d import _broadcast_bmm


class StressVP(nn.Module):
    def __init__(self, device, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        self.camera = PerspectiveCameras(focal_length = torch.tensor([1]).float(), device = device)

    def forward(self, angles, batch):

        elevation = angles[:, 0]
        azimuth = angles[:, 1]
        curr_device = angles.device

        node_pos = batch.init_pos

        # get batched transformation matrices based on batched elevations and azimuths
        R, T = look_at_view_transform(1, elevation.float(), azimuth.float(), device = curr_device)

        res_tf3d_obj = self.camera.get_full_projection_transform(R=R.float(),T=T.float(), device = curr_device)
        # transform_matrix = self.camera.get_full_projection_transform(R=R.float(),T=T.float(), device = curr_device).get_matrix()

        # normally we would do transform_matrix.get_matrix() to acquire the batched transformation matrices
        # however, in pytorch3d this seems to not work for batches larger than size 2. I don't think this has been implemented yet
        # so we'll write it ourselves according to the get_matrix() function in transform3d.py

        composed_matrix = res_tf3d_obj._matrix.clone()
        n_in_batch = R.shape[0]
        other_matrix = res_tf3d_obj._transforms[0].get_matrix()
        composed_matrix = _broadcast_bmm(composed_matrix, other_matrix)
        other_matrix = torch.tensor([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 0., 1.],
                                     [0., 0., 1., 0.]]).float().to(curr_device)
        other_matrix = other_matrix.repeat(n_in_batch, 1, 1).float().to(curr_device)
        transform_matrix = _broadcast_bmm(composed_matrix, other_matrix)

        # get the graph sizes we need this to get the node indices
        batch_n_sizes = batch.n
        batch_indices = torch.repeat_interleave(torch.arange(len(batch_n_sizes)).to(curr_device), batch_n_sizes).to(curr_device)

        # create homogeneous coordinates, needed for projection later
        homogeneous_coords = torch.cat([node_pos, torch.ones(node_pos.size(0), 1).to(curr_device)], dim=1).float()

        # get an nx4x4 transformation matrix
        batched_transforms = transform_matrix[batch_indices].float()  # Nx4x4

        # apply the first transformation matrix to the first node coordinate matrix using batch matrix multiplication
        transformed_homogeneous_coords = torch.bmm(homogeneous_coords.unsqueeze(1), batched_transforms).squeeze(1)
        view = transformed_homogeneous_coords[:, 0:2]

        # you can double check that this is correctly happening with the lines below,
        # first_transformed = torch.matmul(homogeneous_coords[:batch_n_sizes[0]], transform_matrix[0])
        # assert torch.allclose(view[:batch_n_sizes[0]], first_transformed), "Mismatch found in the first batch transformation"

        # scale to range 0,1
        view = view - torch.min(view, axis=0)[0]
        node_pos = view / torch.max(view)

        # now regular method of getting stress according to stress.py from deepgd
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]

        n = len(node_pos)

        # scale eucldean distance with the gtds for fair comparisons
        coords_pairs = torch.hstack((start, end))
        scal_coords = eu.div(d).div((eu.square().div(d.square())).sum()).sum().multiply(coords_pairs)
        eu = (scal_coords[:, 0:2] - scal_coords[:, 2:4]).norm(dim = 1)

        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.full_edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)

        # if divis:
        #     # only n^2 this time since here both pairs (back and forth) are taken
        #     divis_val = ((n ** 2))
        #     graph_stress = graph_stress / divis_val

        # we want to normalize stress output so that the size of the graph has no influence
        # but we want the result not to be bound between 0 and 1 since that could lead to vanishing gradients
        # so we multiply by a 1000 and then normalize
        divis_val = batch.n ** 2 - batch.n
        graph_stress = graph_stress * 1000 / divis_val

        # reduce is here taking the mean of all the stress values of all graphs in the batch
        return graph_stress if self.reduce is None else self.reduce(graph_stress)



