import torch
import torch.nn as nn


def norm_stress_torch_pairs(coords_2d, data_obj, stress_alpha = 2):

    # we take the pairs of nodes to create coordinate pairs, so that we don't compare node x0 to node x0 leading to
    # euclidean distances of 0 which will create nans in the gradient
    idcs_pairs = data_obj.idcs
    coords_pairs = coords_2d[idcs_pairs].flatten(start_dim=1)
    gtds_nn_pairs = data_obj.gtds_pairs

    # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
    weights = gtds_nn_pairs ** -stress_alpha
    weights[weights == float('inf')] = 0
    weights = torch.flatten(weights)
    gtds_nn = torch.flatten(gtds_nn_pairs)

    # calculate the euclidean distances
    eucl_dis = torch.sqrt(torch.sum(((coords_pairs[:, 0:2] - coords_pairs[:, 2:4]) ** 2), 1))
    scal_coords = coords_pairs * (torch.nansum((eucl_dis / gtds_nn) / torch.nansum((eucl_dis ** 2) / (gtds_nn ** 2))))

    eucl_dis_new = torch.sqrt(torch.sum(((scal_coords[:, 0:2] - scal_coords[:, 2:4]) ** 2), 1))

    # compute stress by multiplying the distances with the graph theoretic distances, squaring it, multiplying by the weight factor and then taking the sum
    # 0 is best, 1 is worst, this is flipped later
    ns = torch.mean(weights * (torch.pow((eucl_dis_new - gtds_nn), 2)))

    return ns


def crossings_number(coords_2d, data_obj):

    edges = data_obj.edge_index
    m = int(len(edges.T) / 2)
    # edges in the data object are doubled, once for each direction, so we reduce it by half
    edges = edges[:, :m]

    edge_coords = coords_2d[edges.T]
    edge_coords = edge_coords.reshape(edge_coords.shape[0], 4)

    # do some matrix repetition so that we can get edge pairs later
    matrix_repeated_rows = edge_coords.repeat(m, 1)
    matrix_repeated_cols = edge_coords.repeat_interleave(m, dim=0)

    # mask to filter out self edge comparisons
    indices = torch.arange(m)
    row_indices = indices.repeat_interleave(m)
    col_indices = indices.repeat(m)
    mask = row_indices != col_indices

    filtered_rows = matrix_repeated_rows[mask]
    filtered_cols = matrix_repeated_cols[mask]

    p = torch.cat((filtered_rows, filtered_cols), dim=1)

    cross_bool = cross_pairs(p)

    cnt = torch.sum(cross_bool)
    end_cnt = cnt / 2  # duplicate comparisons done, divide by half to get the actual crossings
    cr_poss = m * (m - 1) / 2
    degrees = torch.tensor(list(dict(data_obj.degrees).values()))
    cr_imp = torch.sum(degrees * (degrees - 1)) / 2

    cn = (end_cnt / (cr_poss - cr_imp))

    return cn


# batched manner of computing crossings, much better for RAM
# simple for loop is 10x faster than DataLoader from torch for some reason
def cross_pairs(p):

    tot_cross_bool = torch.empty((0), dtype = torch.bool)
    batch_size = int(len(p) / 20)

    for i in range(0, len(p), batch_size):
        subp = p[i:(i + batch_size)]
        p1, p2, p3, p4 = subp[:, :2], subp[:, 2:4], subp[:, 4:6], subp[:, 6:]
        a = p2 - p1
        b = p3 - p4
        c = p1 - p3
        ax, ay = a[:, 0], a[:, 1]
        bx, by = b[:, 0], b[:, 1]
        cx, cy = c[:, 0], c[:, 1]

        denom = ay * bx - ax * by
        numer_alpha = by * cx - bx * cy
        numer_beta = ax * cy - ay * cx
        alpha = numer_alpha / denom
        beta = numer_beta / denom

        cross_bool = torch.logical_and(
            torch.logical_and(0 < alpha, alpha < 1),
            torch.logical_and(0 < beta, beta < 1),
        )
        tot_cross_bool = torch.cat((tot_cross_bool, cross_bool))

    return tot_cross_bool


def cross_mlp(coords_2d, data_obj):

    mlp_model = data_obj.mlp_model
    edges = data_obj.edge_index
    m = int(len(edges.T) / 2)
    # edges in the data object are doubled, once for each direction, so we reduce it by half
    edges = edges[:, :m]

    edge_coords = coords_2d[edges.T]
    edge_coords = edge_coords.reshape(edge_coords.shape[0], 4)

    # do some matrix repetition so that we can get edge pairs later
    matrix_repeated_rows = edge_coords.repeat(m, 1)
    matrix_repeated_cols = edge_coords.repeat_interleave(m, dim=0)

    # mask to filter out self edge comparisons
    indices = torch.arange(m)
    row_indices = indices.repeat_interleave(m)
    col_indices = indices.repeat(m)
    mask = row_indices != col_indices

    filtered_rows = matrix_repeated_rows[mask]
    filtered_cols = matrix_repeated_cols[mask]

    p = torch.cat((filtered_rows, filtered_cols), dim=1)

    model_device = next(mlp_model.parameters()).device

    # bcef = nn.BCELoss(reduction = 'mean')
    bcef = nn.MSELoss(reduction = 'mean')
    p = p.to(model_device).float().requires_grad_(True)

    # pred = mlp_model(p)
    # flat_pred = pred.view(-1)
    # target = torch.zeros_like(flat_pred)
    # loss = bcef(flat_pred, target.to(model_device))

    batch_loss = torch.tensor(0, dtype = torch.float64)
    batch_size = int(len(p) / 20)

    for i in range(0, len(p), batch_size):
        pred = mlp_model(p[i:(i + batch_size)])
        flat_pred = pred.view(-1)
        target = torch.zeros_like(flat_pred)
        batch_loss = batch_loss + bcef(flat_pred, target.to(model_device))

    # if we want to check accuracy of the prediction (slow):
    labels = cross_pairs(p)
    # perc = (torch.sum(flat_pred > 0.5) / torch.sum(labels) * 100).item()
    # if perc > 100:
    #     perc = (100 - (perc - 100))
    # print('Accuracy of MLP prediction: ' + str(perc))
    actual_cnt = torch.sum(labels)
    actual_end_cnt = actual_cnt / 2  # duplicate comparisons done, divide by half to get the actual crossings
    cr_poss = m * (m - 1) / 2
    degrees = torch.tensor(list(dict(data_obj.degrees).values()))
    cr_imp = torch.sum(degrees * (degrees - 1)) / 2

    actual_cn = (actual_end_cnt / (cr_poss - cr_imp))

    # the loss above is the cross entropy loss and not the same as the crossing number
    # inc ase you want the crossing number you can uncomment the code below
    cnt = torch.sum(flat_pred > 0.5)
    end_cnt = cnt / 2  # duplicate comparisons done, divide by half to get the actual crossings
    cr_poss = m * (m - 1) / 2
    degrees = torch.tensor(list(dict(data_obj.degrees).values()))
    cr_imp = torch.sum(degrees * (degrees - 1)) / 2

    cn = (end_cnt / (cr_poss - cr_imp))

    return batch_loss, cn, actual_cn


"""
Edge length deviation the current layout

Input
coords:     np.ndarray, a 2xn numpy array containing the x,y coordinates
gtds:       np.ndarray, an nxn numpy array containing the graph theoretic distances (shortest path lengths)

Output
el:         float, the edge length deviation
"""


def edge_lengths_sd_torch(coords_2d, data_obj):

    edges = data_obj.edge_index
    m = int(len(edges.T) / 2)
    # edges in the data object are doubled, once for each direction, so we reduce it by half
    edges = edges[:, :m]

    edge_coords = coords_2d[edges.T]
    edge_coords = edge_coords.reshape(edge_coords.shape[0], 4)

    # calculate the euclidean distances
    eucl_dis = torch.sqrt(torch.sum(((edge_coords[:, 0:2] - edge_coords[:, 2:4]) ** 2), 1))

    mu = torch.mean(eucl_dis)

    # best edge length standard deviation is 0
    el = torch.sqrt(torch.mean((eucl_dis - mu) ** 2))

    return el


def node_occlusion(coords_2d, data_obj, d = None):

    # we take the pairs of nodes to create coordinate pairs, so that we don't compare node x0 to node x0 leading to
    # euclidean distances of 0 which will create nans in the gradient
    idcs_pairs = data_obj.idcs
    coords_pairs = coords_2d[idcs_pairs].flatten(start_dim=1)
    n = len(coords_2d)
    if not d:
        d = 1 / torch.sqrt(torch.tensor(n))

    # calculate the euclidean distances
    eucl_dis = torch.sqrt(torch.sum(((coords_pairs[:, 0:2] - coords_pairs[:, 2:4]) ** 2), 1))
    within = eucl_dis - d
    abs_within = torch.abs(within)
    nn = torch.mean((within - abs_within) / (-2 * d))

    return nn


def node_edge_occlusion(coords_2d, data_obj, d = None):

    edges = data_obj.edge_index
    m = int(len(edges.T) / 2)
    # edges in the data object are doubled, once for each direction, so we reduce it by half
    edges = edges[:, :m]
    n = len(coords_2d)
    if not d:
        d = 1 / torch.sqrt(torch.tensor(n))

    # get the coordinates of node pairs connected by edges (edge coords)
    edge_coords = coords_2d[edges.T]

    # get the x,y coordinates of all edges in separate variables
    x_B = edge_coords[:, 0, 0]
    y_B = edge_coords[:, 0, 1]
    x_C = edge_coords[:, 1, 0]
    y_C = edge_coords[:, 1, 1]

    # do the same thing for the node coordinates
    x_A = coords_2d[:, 0]
    y_A = coords_2d[:, 1]

    # get parts of the vectors BA and BC
    BA_x = x_A[:, None] - x_B[None, :]
    BA_y = y_A[:, None] - y_B[None, :]
    BC_x = x_C[None, :] - x_B[None, :]
    BC_y = y_C[None, :] - y_B[None, :]

    # get the cross product magnitude
    cross_product = torch.abs(BA_x * BC_y - BA_y * BC_x)

    # get the length of the lines, aka the euclidean distances of the edges
    edge_eucl = torch.sqrt(BC_x ** 2 + BC_y ** 2)

    # now we compute the shortest euclidean distances of all the nodes to all the edges
    eucl_dis = cross_product / edge_eucl

    # some nodes are apart of the edges, we do not want to compare these nodes to their own edges
    # so we filter out these node edge comparisons
    node_indices = torch.arange(coords_2d.shape[0])[:, None]  # shape: (n, 1)
    same_as_B = (node_indices == edges[0, :][None, :])  # shape: (n, m)
    same_as_C = (node_indices == edges[1, :][None, :])  # shape: (n, m)

    # we filter out the nodes that are apart of the edges
    eucl_dis[same_as_B | same_as_C] = float('nan')
    # flatten it and then remove nans
    eucl_dis = eucl_dis.flatten()
    # ~inverts the boolean tensor, this we only keep the nonnans
    eucl_dis = eucl_dis[~eucl_dis.isnan()]

    # then follow the same formula as the node node oclusions but we simply consider the closest point on the edge as another node
    # with the same diameter
    within = eucl_dis - d
    abs_within = torch.abs(within)
    ne = torch.mean((within - abs_within) / (-2 * d))

    return ne
