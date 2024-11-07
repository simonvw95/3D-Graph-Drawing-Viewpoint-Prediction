import torch
import pandas as pd
import networkx as nx
import numpy as np
import os
import pickle
from pytorch3d.renderer import PerspectiveCameras
from glob import glob
from tqdm import tqdm

from viewpoint_optimization.basics import get_data
from viewpoint_optimization.metrics_gd_torch import *
from viewpoint_optimization.iterative_resampling import run_iter_resampling
from viewpoint_optimization.uniform_sample import run_uni_sample
from viewpoint_optimization.simulated_annealing import run_sim_annealing
from viewpoint_optimization.genetic_computing import run_genetic_algorithm
from viewpoint_optimization.particle_swarm import run_pso
from viewpoint_optimization.gradient_descent import run_gradient_descent
from viewpoint_optimization.uni_gradient_descent import run_uni_grad_descent
from viewpoint_optimization.uni_gradient_descent2 import run_uni_grad_descent_v2
from viewpoint_optimization.newton_raphson import run_newton_raphson
from viewpoint_optimization.basics import loss_function
from neural_aesthete import CrossingDetector
from utils import rectangular_to_spherical


if __name__ == '__main__':

    device = 'cpu'
    l_bound = torch.tensor([-90, -90])
    u_bound = torch.tensor([90, 90])
    bounds = list(zip(l_bound, u_bound))
    camera1 = PerspectiveCameras(device = device, focal_length = torch.tensor([1]).float().to(device))

    # vary N_max depending on how many function evaluations you want to do
    N_max = 40

    # qm_weights = {'Stress' : 1, 'Edge Length Deviation' : 0.8, 'Node Node Occlusion' : 0.9, 'Node Edge Occlusion' : None, 'Crossing Number' : 1}
    qm_weights = {'Stress': None, 'Edge Length Deviation': None, 'Node Node Occlusion': None, 'Node Edge Occlusion': None, 'Crossing Number': 1}
    qm_functions_map = {'Stress' : norm_stress_torch_pairs, 'Edge Length Deviation' : edge_lengths_sd_torch, 'Node Node Occlusion' : node_occlusion, 'Node Edge Occlusion' : node_edge_occlusion, 'Crossing Number' : [crossings_number, cross_mlp]}

    qm_funcs_norm = {}
    qm_funcs_grad = {}
    for key in qm_weights:
        qm = qm_functions_map[key]
        # the crossing number has two functions
        if isinstance(qm, list):
            # first one is the discrete computation
            qm_funcs_norm[qm[0]] = qm_weights[key]
            qm_funcs_grad[qm[1]] = qm_weights[key]
        else:
            qm_funcs_norm[qm] = qm_weights[key]
            qm_funcs_grad[qm] = qm_weights[key]

    qm_function_name = ''
    for key in qm_funcs_norm:
        if qm_funcs_norm[key]:
            qm_function_name += key.__name__ + str(qm_funcs_norm[key])

    if qm_weights['Crossing Number']:
        mlp_model = CrossingDetector().to(device).eval()
        mlp_model.load_state_dict(torch.load('evaluations/mlp_cross.pt', map_location=device))

    sorted_file_names = ['sierpinski3d', 'GD96_c', 'L', 'can_96', 'dwt_1005']
    # sorted_file_names = ['GD96_c']
    # graphs for cn
    sorted_file_names = ['GD96_c', 'mesh1em6', 'gridaug', 'grafo6975', 'grafo10230']

    print('Computing best viewpoint using various strategies')
    qm_count = 0
    for key in qm_weights:
        if qm_weights[key]:
            print('Quality metric function: ' + key + ' | with weight: ' + str(qm_weights[key]))
            qm_count += 1

    # set write to false because for the manual comparison we don't want to keep track of the loss progression, only the end result
    write = False

    # in the following experiments N_max is always the maximum number of function evaluations (calls to the loss function)
    print('-----------------------------------------------------')
    print('-----------------------------------------------------')
    print('-----------------------------------------------------')
    print('Maximum number of function evaluations: ' + str(N_max))
    graph_results = {}
    for key in sorted_file_names:
        graph_results[key] = {}

    graph_cnt = 0

    spec_dir = 'evaluations/drawings/' + qm_function_name + '-' + str(N_max) + '/'

    # loop over every graph
    for dataset_name in tqdm(sorted_file_names, position = 0, leave = True):

        # create the directory if it doesn't exist yet
        if not os.path.isdir(spec_dir):
            os.makedirs(spec_dir)

        # check if we already have a pkl file with the results so we don't redo computations
        pkl_filename = 'evaluations/drawings/' + qm_function_name + '-' + str(N_max) + '.pkl'
        if os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as handle:
                graph_results = pickle.load(handle)

        print('-----------------------------------------------------')
        print('Graph number: ' + str(graph_cnt) + '/' + str(len(sorted_file_names)) + ' | Starting with graph: ' + dataset_name)
        # create the names of the edgelist file and the metric file
        input_file = glob(f'data/{dataset_name}/*-src.csv')[0]
        layout_file_3d = "layouts/" + dataset_name + "-FA2-3d.csv"
        coords_3d = pd.read_csv(layout_file_3d, sep=';').to_numpy()

        # get the graph object and edges
        df = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "-src.csv", sep=';', header=0)
        G = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        G = nx.convert_node_labels_to_integers(G)
        # some graphs sometimes have selfloops
        G.remove_edges_from(nx.selfloop_edges(G))
        gtds_np = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "-gtds.csv", sep=';', header=0).to_numpy()

        data_obj = get_data(G, coords_3d, gtds_np=gtds_np)

        data_obj.N_max = N_max
        data_obj.name = dataset_name
        data_obj.qm_function_name = qm_function_name

        if qm_weights['Crossing Number']:
            data_obj.mlp_model = mlp_model

        n_rep = 5
        all_res = {}
        print('Finished getting graph data, starting strategies now')
        # 5 repetitions in the experiment to average out some randomness
        for i in range(n_rep):
            data_obj.iteration = i

            all_res = {}

            print('-----------------------------------------------------')
            print('Iteration: ' + str(i))

            key_name = 'Uniform Sample'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'uni_sample'
                sample_uni_results = run_uni_sample(N = N_max, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, bounds = bounds, write = write)
                best_val = 1 - sample_uni_results['best_val'].item()

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': rectangular_to_spherical(np.expand_dims(sample_uni_results['best_angle'], axis = 0))[0]}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': rectangular_to_spherical(np.expand_dims(sample_uni_results['best_angle'], axis = 0))[0]}

            # Sample division heuristic
            key_name = 'Iterative Resampling'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'sample_div'
                # number of iterations depends on how many layers we have
                divisions = 2
                if N_max < 40:
                    divisions = 1
                # the total number of function evaluations for sample division is the first uniform sampling pass and then divisions * that sampling pass
                # so if N_max = 100, and there are 4 divisions, then first pass should be 20, then the following 4 divisions should be 20 each, hence divisions + 1 in computation
                N = int(N_max / (divisions + 1))
                sample_div_results = run_iter_resampling(N = N, divisions = divisions, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, bounds = bounds, write = write)
                best_val = 1 - sample_div_results['best_val'].item()

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': rectangular_to_spherical(np.expand_dims(sample_div_results['best_angle'], axis = 0))[0]}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': rectangular_to_spherical(np.expand_dims(sample_div_results['best_angle'], axis = 0))[0]}

            # Simulated Annealing
            key_name = 'Simulated Annealing (1)'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'sim_anneal1'
                # simulated annealing has its own built in max number of evaluations parameter
                # random viewpoint start
                curr_viewpoint = np.array([(np.random.rand(1) - 0.5) * 180, (np.random.rand(1) - 0.5) * 180]).T[0]
                temp = 5230

                sim_ann_results = run_sim_annealing(N = N_max, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, curr_viewpoint = curr_viewpoint, temp = temp, write = write)
                if torch.is_tensor(sim_ann_results.fun):
                    best_val = 1 - sim_ann_results.fun.item()
                else:
                    best_val = 1 - sim_ann_results.fun

                # if we already have a result then compare current result to it
                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss' : best_val, 'vwp' : np.append(sim_ann_results.x, 1)}
                else:
                    all_res[key_name] = {'loss' : best_val, 'vwp' : np.append(sim_ann_results.x, 1)}

            # different viewpoints start
            key_name = 'Simulated Annealing (5)'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'sim_anneal5'
                curr_viewpoints = np.array([[0, 0], [45, 0], [0, 45], [0, -45], [-45, 0]], dtype=np.float32)
                mult_results_vals = []
                mult_results_vwps = []
                N = int(N_max / 5)
                temp = 5230 * 5

                for vwp in curr_viewpoints:
                    sim_ann_results = run_sim_annealing(N = N, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, curr_viewpoint = vwp, temp = temp, write = write)
                    if torch.is_tensor(sim_ann_results.fun):
                        best_val = 1 - sim_ann_results.fun.item()
                    else:
                        best_val = 1 - sim_ann_results.fun
                    mult_results_vals.append(best_val)
                    mult_results_vwps.append(sim_ann_results.x)
                best_idx = np.argmax(mult_results_vals)
                best_val = mult_results_vals[best_idx]
                best_vwp = mult_results_vwps[best_idx]

                # if we already have a result then compare current result to it
                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss' : best_val, 'vwp' : np.append(best_vwp, 1)}
                else:
                    all_res[key_name] = {'loss' : best_val, 'vwp' : np.append(best_vwp, 1)}

            # Genetic Algorithm
            key_name = 'Differential Evolution'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'genetic_alg'
                # the total number of evaluations is equal to (max_iter + 1) * pop_size * P
                # so we set max number of iterations using the following calculation
                pop_size = 10
                P = 2  # number of unique bounds
                N = int(N_max / (pop_size * P) - 1)
                gen_alg_results = run_genetic_algorithm(N = N, popsize = pop_size, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, write = write)

                if torch.is_tensor(gen_alg_results.fun):
                    best_val = 1 - gen_alg_results.fun.item()
                else:
                    best_val = 1 - gen_alg_results.fun

                # if we already have a result then compare current result to it
                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss' : best_val, 'vwp' : np.append(gen_alg_results.x, 1)}
                else:
                    all_res[key_name] = {'loss' : best_val, 'vwp' : np.append(gen_alg_results.x, 1)}

            # Particle Swarm Optimization
            key_name = 'Particle Swarm Optimization'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'pso'
                # pso is similar to genetic algorithm in that it uses particles/population
                particles = 10
                N = int(N_max / particles)
                pso_results = run_pso(N = N, particles = particles, l_bound = np.array(l_bound), u_bound = np.array(u_bound), data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, write = write)

                if torch.is_tensor(pso_results[0]):
                    best_val = 1 - pso_results[0].item()
                else:
                    best_val = 1 - pso_results[0]

                # if we already have a result then compare current result to it
                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': np.append(pso_results[1], 1)}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': np.append(pso_results[1], 1)}

            # Gradient Descent
            # single viewpoint
            key_name = 'Gradient Descent (1)'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'grad_desc1'
                curr_viewpoint = np.array([(np.random.rand(1) - 0.5) * 180, (np.random.rand(1) - 0.5) * 180]).T[0]
                lr = 0.15
                grad_desc_results = run_gradient_descent(curr_viewpoint = curr_viewpoint, data_obj = data_obj, N = N_max, lr = lr, device = device, qm_function = qm_funcs_grad, bounds = bounds, write = write)
                # instead of taking the value from grad desc results we compute it with the viewpiont angles
                # since the approximation of the crossing number does not return the actual crossing number
                best_val = 1 - loss_function(grad_desc_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, write).item()

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': grad_desc_results['best_vwp']}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': grad_desc_results['best_vwp']}

            # multiple viewpoint start
            key_name = 'Gradient Descent (5)'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'grad_desc5'
                curr_viewpoints = np.array([[0, 0], [45, 0], [0, 45], [0, -45], [-45, 0]], dtype=np.float32)
                lr = 0.5
                N_grad = int(N_max / 5)
                mult_results_vals = []
                mult_results_vwps = []
                for vwp in curr_viewpoints:
                    grad_desc_results = run_gradient_descent(curr_viewpoint = vwp, data_obj = data_obj, N = N_grad, lr = lr, device = device, qm_function = qm_funcs_grad, bounds = bounds, write = write)

                    best_val = 1 - loss_function(grad_desc_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, write).item()
                    mult_results_vals.append(best_val)
                    mult_results_vwps.append(grad_desc_results['best_vwp'])

                best_idx = np.argmax(mult_results_vals)
                best_val = mult_results_vals[best_idx]
                best_vwp = mult_results_vwps[best_idx]

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': best_vwp}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': best_vwp}

            # Uniform Gradient Descent, doing gradient descent from 10 different starts
            key_name = 'Uniform Gradient Descent V1'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'uni_grad'
                N_start = 10
                N_grad = int(N_max / N_start)
                lr = 0.45
                uni_grad_res = run_uni_grad_descent(N_start = N_start, N_grad = N_grad, lr = lr, data_obj = data_obj, qm_function = qm_funcs_grad, bounds = bounds, write = write)
                best_val = 1 - loss_function(uni_grad_res['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, write).item()

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': uni_grad_res['best_vwp']}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': uni_grad_res['best_vwp']}

            # Uniform Gradient Descent version 2, doing gradient descent from the best viewpoint out of 10 starting viewpoints
            key_name = 'Uniform Gradient Descent V2'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'uni_gradv2'
                N_start = 10
                N_grad = int(N_max - N_start)
                lr = 0.15
                uni_grad_res = run_uni_grad_descent_v2(N_start = N_start, N_grad = N_grad, lr = lr, data_obj = data_obj, qm_function_norm = qm_funcs_norm, qm_function_grad = qm_funcs_grad, bounds = bounds, camera = camera1, write = write)
                best_val = 1 - loss_function(uni_grad_res['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, write).item()

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': uni_grad_res['best_vwp']}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': uni_grad_res['best_vwp']}

            # Newton-Raphson
            # starting with one viewpoint
            key_name = 'Newton-Raphson (1)'
            if key_name not in graph_results[dataset_name]:
                data_obj.strategy = 'newton_raphson1'
                curr_viewpoint = np.array([(np.random.rand(1) - 0.5) * 180, (np.random.rand(1) - 0.5) * 180]).T[0]
                newt_raphs_results = run_newton_raphson(N = N_max, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_grad,
                                                        camera = camera1, curr_viewpoint = curr_viewpoint)
                data_obj.strategy = 'test'  # change the strategy name because if it is newton raphson then we get the gradient back which we don't want here
                best_val = 1 - loss_function(newt_raphs_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, write).item()

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': newt_raphs_results['best_vwp']}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': newt_raphs_results['best_vwp']}

            # starting with multiple viewpoints
            key_name = 'Newton-Raphson (5)'
            if key_name not in graph_results[dataset_name]:
                curr_viewpoints = np.array([[0, 0], [45, 0], [0, 45], [0, -45], [-45, 0]], dtype=np.float32)
                N_grad = int(N_max / 5)
                mult_results_vals = []
                mult_results_vwps = []
                for vwp in curr_viewpoints:
                    data_obj.strategy = 'newton_raphson5'
                    newt_raphs_results = run_newton_raphson(N = N_grad, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_grad, camera = camera1, curr_viewpoint = vwp)
                    data_obj.strategy = 'test'  # change the strategy name because if it is newton raphson then we get the gradient back which we don't want here
                    best_val = 1 - loss_function(newt_raphs_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, write).item()
                    mult_results_vals.append(best_val)
                    mult_results_vwps.append(newt_raphs_results['best_vwp'])

                best_idx = np.argmax(mult_results_vals)
                best_val = mult_results_vals[best_idx]
                best_vwp = mult_results_vwps[best_idx]

                if key_name in all_res:
                    if best_val > all_res[key_name]['loss']:
                        all_res[key_name] = {'loss': best_val, 'vwp': best_vwp}
                else:
                    all_res[key_name] = {'loss': best_val, 'vwp': best_vwp}

        graph_cnt += 1

        print('Doing Ground Truth')
        # Ground Truth
        key_name = 'Ground Truth'
        if key_name not in graph_results[dataset_name]:
            data_obj.strategy = 'sample_div'
            # number of iterations depends on how many layers we have
            divisions = 1
            # the total number of function evaluations for sample division is the first uniform sampling pass and then divisions * that sampling pass
            # so if N_max = 100, and there are 4 divisions, then first pass should be 20, then the following 4 divisions should be 20 each, hence divisions + 1 in computation
            N = int(10000 / (divisions + 1))
            ground_truth_results = run_iter_resampling(N = N, divisions = divisions, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, bounds = bounds, write = write)
            best_val = 1 - ground_truth_results['best_val'].item()

            all_res[key_name] = {'loss': best_val, 'vwp': rectangular_to_spherical(np.expand_dims(ground_truth_results['best_angle'], axis=0))[0]}

        # now we put all the newly computed results in the dictionary and overwrite the old pickle file with the new results
        for kn in all_res:
            if kn not in graph_results[dataset_name]:
                graph_results[dataset_name][kn] = all_res[kn]

        with open(pkl_filename, 'wb') as handle:
            pickle.dump(graph_results, handle)
