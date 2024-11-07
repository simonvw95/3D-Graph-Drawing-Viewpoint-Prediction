import networkx as nx
import torch
import pandas as pd
import os
import pickle
import torch_geometric as pyg
from tqdm import tqdm
from pytorch3d.renderer import (PerspectiveCameras)

import deepgd as dgd


# convert the graph objects to a data object from pytorch geometric used in DeepGD
def convert(G, coords):
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    init_pos = torch.tensor(coords)

    full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
    raw_edge_index = pyg.utils.to_undirected(torch.tensor(list(G.edges)).T)
    full_edge_index, d = pyg.utils.remove_self_loops(*pyg.utils.to_undirected(
        torch.tensor(full_edges).T, torch.tensor(attr_d)
    ))
    k = 1 / d ** 2
    full_edge_attr = torch.stack([d, k], dim=-1)
    return pyg.data.Data(
        G=G,
        x=init_pos,
        init_pos=init_pos,
        edge_index=full_edge_index,
        edge_attr=full_edge_attr,
        raw_edge_index=raw_edge_index,
        full_edge_index=full_edge_index,
        full_edge_attr=full_edge_attr,
        d=d,
        n=G.number_of_nodes(),
        m=G.number_of_edges()
    )


# load in the device
device = "cuda"
for backend, device_name in {
    torch.backends.mps: "mps",
    torch.cuda: "cuda",
}.items():
    if backend.is_available():
        device = device_name

# manually set to cpu due to memory issues
device = 'cpu'

# loop over a whole set of graphs
all_datasets = os.listdir('../data/')
sizes = {}
for i in all_datasets:
    sizes[i] = os.path.getsize('../data/' + i + '/' + i + '-gtds.csv')
sorted_file_names = sorted(sizes, key=sizes.get)
# all
# sorted_file_names = sorted_file_names[:45]
# only five for manual comparison
# sorted_file_names = ['sierpinski3d', 'GD96_c', 'L', 'can_96', 'dwt_1005']
# only one at a time in case of memory issues
sorted_file_names = ['dwt_1005']

model_test = dgd.DeepGD().to(device).float()
model_test.load_state_dict(torch.load('../saved_models/03-09_20-03_vwp_pred_stress.pt'))
model_test.eval()

# open the pickle file if it already exists
if os.path.isfile('../saved_models/deepgd_results_manual5.pkl'):
    with open('../saved_models/deepgd_results_manual5.pkl', 'rb') as handle:
        results = pickle.load(handle)
else:
    results = {}

# loop over all graphs
for graph_name in tqdm(sorted_file_names):

    # get the 3d cords
    layout_file_3d = "../layouts/" + graph_name + "-FA2-3d.csv"
    camera1 = PerspectiveCameras(device = device, focal_length = torch.tensor([1]).float().to(device))
    coords = pd.read_csv(layout_file_3d, sep=';').to_numpy()

    # get the graph object and edges
    df = pd.read_csv("../data/" + graph_name + "/" + graph_name + "-src.csv", sep=';', header=0)
    G = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
    G = nx.convert_node_labels_to_integers(G)

    # getting the graph in the pytorch geometric Data format
    testgraph = convert(G, coords).to(device)
    # need to put it in a batch for the loss function to work properly
    testloader = pyg.loader.DataLoader([testgraph], batch_size=1, shuffle=False)

    for batch in testloader:
        pred_angles = model_test(batch)

    stress_func = dgd.StressVP(device)
    # divide by 1000 because we i
    stress_val = 1 - (stress_func.forward(pred_angles, batch.to(device)).item() / 1000)
    results[graph_name] = {'DeepGD (norm)' : {'loss' : stress_val, 'vwp' : torch.concatenate((pred_angles[0], torch.tensor([1])))}}
    print('Best quality metric value found: ' + str(stress_val) + ' for graph: ' + graph_name)

with open('../saved_models/deepgd_results_manual5.pkl', 'wb') as handle:
    pickle.dump(results, handle)
