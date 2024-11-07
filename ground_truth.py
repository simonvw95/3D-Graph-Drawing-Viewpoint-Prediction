import pandas as pd
import networkx as nx
import os
import pickle

from pytorch3d.renderer import PerspectiveCameras
from glob import glob

from viewpoint_optimization.basics import get_data
from viewpoint_optimization.metrics_gd_torch import *
from viewpoint_optimization.iterative_resampling import run_iter_resampling


if __name__ == '__main__':

    device = 'cpu'
    l_bound = torch.tensor([-90, -90])
    u_bound = torch.tensor([90, 90])
    bounds = list(zip(l_bound, u_bound))
    camera1 = PerspectiveCameras(device = device, focal_length = torch.tensor([1]).float().to(device))

    # qm_weights = {'Stress': 1, 'Edge Length Deviation': 0.8, 'Node Node Occlusion': 0.9, 'Node Edge Occlusion': None, 'Crossing Number': 1}
    qm_weights = {'Stress': None, 'Edge Length Deviation': None, 'Node Node Occlusion': None, 'Node Edge Occlusion': None, 'Crossing Number': 1}
    qm_functions_map = {'Stress' : norm_stress_torch_pairs, 'Edge Length Deviation' : edge_lengths_sd_torch, 'Node Node Occlusion' : node_occlusion, 'Node Edge Occlusion' : node_edge_occlusion, 'Crossing Number' : crossings_number}

    qm_funcs_norm = {}
    for key in qm_weights:
        qm = qm_functions_map[key]
        qm_funcs_norm[qm] = qm_weights[key]

    qm_function_name = ''
    for key in qm_funcs_norm:
        if qm_funcs_norm[key]:
            qm_function_name += key.__name__ + str(qm_funcs_norm[key])

    all_datasets = os.listdir('data/')
    sizes = {}
    for i in all_datasets:
        sizes[i] = os.path.getsize('data/' + i + '/' + i + '-gtds.csv')

    sorted_file_names = sorted(sizes, key=sizes.get)
    # main subset of graphs for every metric except crossing number
    sorted_file_names = sorted_file_names[:45]

    # for crossing number we manually remove lesmis and can_96 since crashes occurred for these graphs inexplicably
    # sorted_file_names.remove('lesmis')
    # sorted_file_names.remove('can_96')
    # # we select only 19 graphs for crossing number
    # sorted_file_names = sorted_file_names[:19]
    sorted_file_names = ['grafo10231']

    print('Computing best viewpoint using various strategies')
    print('Quality metric function: ' + str(qm_function_name))

    # in the following experiments N_max is always the maximum number of function evaluations (calls to the loss function)
    N_max = 10000
    print('-----------------------------------------------------')
    print('Maximum number of function evaluations: ' + str(N_max))
    graph_results = {}

    for dataset_name in sorted_file_names:
        print('-----------------------------------------------------')
        print('Starting with graph: ' + dataset_name)
        # create the names of the edgelist file and the metric file
        input_file = glob(f'data/{dataset_name}/*-src.csv')[0]
        layout_file_3d = "layouts/" + dataset_name + "-FA2-3d.csv"
        coords = pd.read_csv(layout_file_3d, sep=';').to_numpy()

        # get the graph object and edges
        df = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "-src.csv", sep=';', header=0)
        G = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        G = nx.convert_node_labels_to_integers(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        gtds_np = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "-gtds.csv", sep=';', header=0).to_numpy()

        data_obj = get_data(G, coords, gtds_np = gtds_np)
        data_obj.strategy = 'sample_div'
        divisions = 1
        # the total number of function evaluations for sample division is the first uniform sampling pass and then divisions * that sampling pass
        # so if N_max = 100, and there are 4 divisions, then first pass should be 20, then the following 4 divisions should be 20 each, hence divisions + 1 in computation
        N = int(N_max / (divisions + 1))

        sample_div_results = run_iter_resampling(N = N, divisions = divisions, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, bounds = bounds, write = False)
        best_val = 1 - sample_div_results['best_val'].item()
        worst_val = 1 - sample_div_results['worst_val'].item()
        graph_results[dataset_name] = {'Ground Truth' : {'best_loss' : best_val, 'worst_loss' : worst_val, 'best_vwp' : sample_div_results['best_angle'], 'worst_vwp' : sample_div_results['worst_angle']}}

    with open('evaluations/ground_truth-' + qm_function_name + '.pkl', 'wb') as handle:
        pickle.dump(graph_results, handle)
