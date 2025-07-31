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


def file_exists(data_obj):
    full_name = data_obj.qm_function_name + '-' + data_obj.name + '-' + data_obj.strategy + '-' + str(data_obj.N_max)
    full_name += '-i' + str(data_obj.iteration)

    if os.path.isfile('evaluations/results/' + data_obj.qm_function_name + '/' + full_name + '.txt'):
        return True
    else:
        return False


# some strategies use parallel runs but are evaluated in sequences, so we write 'start' to indicate a new sequential run
# that way in post processing we can put them side by side
def init_file(data_obj):

    full_name = data_obj.qm_function_name + '-' + data_obj.name + '-' + data_obj.strategy + '-' + str(data_obj.N_max)
    full_name += '-i' + str(data_obj.iteration)

    with open('evaluations/results/' + data_obj.qm_function_name + '/' + full_name + '.txt', "a") as file:
        file.write("start\n")
        file.close()


if __name__ == '__main__':

    device = 'cpu'
    l_bound = torch.tensor([-90, -90])
    u_bound = torch.tensor([90, 90])
    bounds = list(zip(l_bound, u_bound))
    camera1 = PerspectiveCameras(device = device, focal_length = torch.tensor([1]).float().to(device))

    # qm_weights = {'Stress' : 1, 'Edge Length Deviation' : 0.8, 'Node Node Occlusion' : 0.9, 'Node Edge Occlusion' : None, 'Crossing Number' : 1}
    qm_weights = {'Stress': 1, 'Edge Length Deviation': None, 'Node Node Occlusion': None, 'Node Edge Occlusion': None, 'Crossing Number': None}
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

    if not os.path.isdir('evaluations/results/' + qm_function_name + '/'):
        os.makedirs('evaluations/results/' + qm_function_name + '/')

    if qm_weights['Crossing Number']:
        mlp_model = CrossingDetector().to(device).eval()
        mlp_model.load_state_dict(torch.load('evaluations/mlp_cross.pt', map_location=device))

    # load the dataset and sort by size
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

    print('Computing best viewpoint using various strategies')
    qm_count = 0
    for key in qm_weights:
        if qm_weights[key]:
            print('Quality metric function: ' + key + ' | with weight: ' + str(qm_weights[key]))
            qm_count += 1

    # max number of function evaluations
    N_max = 500

    # set write to True if we want to keep track of the loss progression via writing to .txt files
    write = True
    # in the following experiments N_max is always the maximum number of function evaluations (calls to the loss function)
    print('-----------------------------------------------------')
    print('-----------------------------------------------------')
    print('-----------------------------------------------------')
    print('Maximum number of function evaluations: ' + str(N_max))
    graph_results = {}

    graph_cnt = 0
    for dataset_name in tqdm(sorted_file_names, position = 0, leave = True):

        print('-----------------------------------------------------')
        print('Graph number: ' + str(graph_cnt) + '/' + str(len(sorted_file_names)) + ' | Starting with graph: ' + dataset_name)
        # create the names of the edgelist file and the metric file
        input_file = glob(f'data/{dataset_name}/*-src.csv')[0]
        layout_file_3d = "layouts/" + dataset_name + "-FA2-3d.csv"
        coords = pd.read_csv(layout_file_3d, sep=';').to_numpy()

        # get the graph object and edges
        df = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "-src.csv", sep=';', header=0)
        G = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        G = nx.convert_node_labels_to_integers(G)
        # some graphs sometimes have selfloops
        G.remove_edges_from(nx.selfloop_edges(G))
        gtds_np = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "-gtds.csv", sep=';', header=0).to_numpy()

        data_obj = get_data(G, coords, gtds_np=gtds_np)

        data_obj.N_max = N_max
        data_obj.name = dataset_name
        data_obj.qm_function_name = qm_function_name

        if qm_weights['Crossing Number']:
            data_obj.mlp_model = mlp_model

        # 5 repetitions in the experiment to average out some randomness
        for i in range(5):
            data_obj.iteration = i

            all_res = {}

            print('-----------------------------------------------------')
            print('Finished getting graph data, starting strategies now. Iteration: ' + str(i))

            # Simulated Annealing
            data_obj.strategy = 'sim_anneal1'
            # simulated annealing has its own built in max number of evaluations parameter
            # random viewpoint start
            curr_viewpoint = np.array([(np.random.rand(1) - 0.5) * 180, (np.random.rand(1) - 0.5) * 180]).T[0]
            temp = 5230
            if not file_exists(data_obj):
                sim_ann_results = run_sim_annealing(N = N_max, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, curr_viewpoint = curr_viewpoint, temp = temp, write = write)
                if torch.is_tensor(sim_ann_results.fun):
                    best_val = round(1 - sim_ann_results.fun.item(), 6)
                else:
                    best_val = round(1 - sim_ann_results.fun, 6)
                all_res['Simulated Annealing (1)'] = best_val

            # different viewpoints start
            data_obj.strategy = 'sim_anneal5'
            curr_viewpoints = np.array([[0, 0], [45, 0], [0, 45], [0, -45], [-45, 0]], dtype=np.float32)
            mult_results = []
            N = int(N_max / 5)
            temp = 5230 * 5
            if not file_exists(data_obj):
                for vwp in curr_viewpoints:
                    init_file(data_obj)
                    sim_ann_results = run_sim_annealing(N = N, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, curr_viewpoint = vwp, temp = temp, write = write)
                    if torch.is_tensor(sim_ann_results.fun):
                        best_val = round(1 - sim_ann_results.fun.item(), 6)
                    else:
                        best_val = round(1 - sim_ann_results.fun, 6)
                    mult_results.append(best_val)
                best_val = max(mult_results)
                all_res['Simulated Annealing (5)'] = best_val

            # Genetic Algorithm
            data_obj.strategy = 'genetic_alg'
            # the total number of evaluations is equal to (max_iter + 1) * pop_size * P
            # so we set max number of iterations using the following calculation
            pop_size = 10
            P = 2  # number of unique bounds
            N = int(N_max / (pop_size * P) - 1)
            if not file_exists(data_obj):
                gen_alg_results = run_genetic_algorithm(N = N, popsize = pop_size, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, write = write)

                if torch.is_tensor(gen_alg_results.fun):
                    best_val = round(1 - gen_alg_results.fun.item(), 6)
                else:
                    best_val = round(1 - gen_alg_results.fun, 6)

                all_res['Genetic Algorithm'] = best_val

            # Particle Swarm Optimization
            data_obj.strategy = 'pso'
            # pso is similar to genetic algorithm in that it uses particles/population
            particles = 10
            N = int(N_max / particles)
            if not file_exists(data_obj):
                pso_results = run_pso(N = N, particles = particles, l_bound = np.array(l_bound), u_bound = np.array(u_bound), data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, write = write)
                if torch.is_tensor(pso_results[0]):
                    best_val = round(1 - pso_results[0].item(), 6)
                else:
                    best_val = round(1 - pso_results[0], 6)

                all_res['Particle Swarm Optimization'] = best_val

            # Gradient Descent
            # single viewpoint
            data_obj.strategy = 'grad_desc1'
            curr_viewpoint = np.array([(np.random.rand(1) - 0.5) * 180, (np.random.rand(1) - 0.5) * 180]).T[0]
            lr = 0.15
            if not file_exists(data_obj):
                grad_desc_results = run_gradient_descent(curr_viewpoint = curr_viewpoint, data_obj = data_obj, N = N_max, lr = lr, device = device, qm_function = qm_funcs_grad, bounds = bounds, write = write)
                # instead of taking the value from grad desc results we compute it with the viewpoint angles
                # since the approximation of the crossing number does not return the actual crossing number
                best_val = round(1 - loss_function(grad_desc_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, False).item(), 6)
                all_res['Gradient Descent (1)'] = best_val

            # multiple viewpoint start
            data_obj.strategy = 'grad_desc5'
            curr_viewpoints = np.array([[0, 0], [45, 0], [0, 45], [0, -45], [-45, 0]], dtype=np.float32)
            lr = 0.5
            N_grad = int(N_max / 5)
            mult_results = []
            if not file_exists(data_obj):
                for vwp in curr_viewpoints:
                    init_file(data_obj)
                    grad_desc_results = run_gradient_descent(curr_viewpoint = vwp, data_obj = data_obj, N = N_grad, lr = lr, device = device, qm_function = qm_funcs_grad, bounds = bounds, write = write)

                    best_val = round(1 - loss_function(grad_desc_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, False).item(), 6)
                    mult_results.append(best_val)
                best_val = max(mult_results)
                all_res['Gradient Descent (5)'] = best_val

            # Uniform Gradient Descent, doing gradient descent from 10 different starts
            data_obj.strategy = 'uni_grad'
            N_start = 10
            N_grad = int(N_max / N_start)
            lr = 0.45
            if not file_exists(data_obj):
                uni_grad_res = run_uni_grad_descent(N_start = N_start, N_grad = N_grad, lr = lr, data_obj = data_obj, qm_function = qm_funcs_grad, bounds = bounds, write = write)
                best_val = round(1 - loss_function(uni_grad_res['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, False).item(), 6)
                all_res['Uniform + Gradient Descent'] = best_val


            # Uniform Gradient Descent version 2, doing gradient descent from the best viewpoint out of 10 starting viewpoints
            data_obj.strategy = 'uni_gradv2'
            N_start = 10
            N_grad = int(N_max - N_start)
            lr = 0.15
            if not file_exists(data_obj):
                uni_grad_res = run_uni_grad_descent_v2(N_start = N_start, N_grad = N_grad, lr = lr, data_obj = data_obj, qm_function_norm = qm_funcs_norm, qm_function_grad = qm_funcs_grad, bounds = bounds, camera = camera1, write = write)
                best_val = round(1 - loss_function(uni_grad_res['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, False).item(), 6)
                all_res['Uniform + Gradient Descent Best'] = best_val


            # Newton-Raphson
            # starting with one viewpoint
            data_obj.strategy = 'newton_raphson1'
            curr_viewpoint = np.array([(np.random.rand(1) - 0.5) * 180, (np.random.rand(1) - 0.5) * 180]).T[0]
            if not file_exists(data_obj):
                newt_raphs_results = run_newton_raphson(N = N_max, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_grad,
                                                        camera = camera1, curr_viewpoint = curr_viewpoint, write = write)
                # rename strategy so that we don't compute the gradient here since that is impossible with the discrete version of crossing number
                data_obj.strategy = 'irrelevant'
                best_val = round(1 - loss_function(newt_raphs_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, False).item(), 6)

                all_res['Newton-Raphson (1)'] = best_val

            # starting with multiple viewpoints
            data_obj.strategy = 'newton_raphson5'
            curr_viewpoints = np.array([[0, 0], [45, 0], [0, 45], [0, -45], [-45, 0]], dtype=np.float32)
            N_grad = int(N_max / 5)
            mult_results = []
            if not file_exists(data_obj):
                for vwp in curr_viewpoints:
                    data_obj.strategy = 'newton_raphson5'
                    init_file(data_obj)
                    newt_raphs_results = run_newton_raphson(N = N_grad, bounds = bounds, data_obj = data_obj, qm_function = qm_funcs_grad, camera = camera1, curr_viewpoint = vwp, write = write)
                    # rename strategy so that we don't compute the gradient here
                    data_obj.strategy = 'irrelevant'
                    best_val = round(1 - loss_function(newt_raphs_results['best_vwp'][:2], data_obj, qm_funcs_norm, camera1, False).item(), 6)
                    mult_results.append(best_val)
                    best_val = max(mult_results)

                all_res['Newton-Raphson (5)'] = best_val

            # if the dictionary is filled then print some results
            if all_res:
                print('All strategies done!')
                best_strategy = sorted(all_res, key = all_res.get, reverse = True)[0]
                print('Best strategy was: ' + best_strategy + ' with a value of: ' + str(all_res[best_strategy]))
                graph_results[dataset_name] = all_res
        graph_cnt += 1

        # these two methods should have zero randomness, thus they do not need to be averaged
        # but, depending on the number of function evaluations they do change in contrast to the other strategies
        # thats why we simply run these a certain number of times for different numbers function evaluations
        data_obj.iteration = 0
        print('-----------------------------------------------------')
        print('Finished all Strategies, now doing uniform sample and sample division')
        for N_max in range(10, 510, 10):
            data_obj.N_max = N_max
            # Uniform sample heuristic
            # number of iterations is equal to N_max
            data_obj.strategy = 'uni_sample'
            if not file_exists(data_obj):
                sample_uni_results = run_uni_sample(N = N_max, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, bounds = bounds, write = write)
                best_val = round(1 - sample_uni_results['best_val'].item(), 6)
                all_res['Uniform Sampling'] = best_val

            # iterative resampling heuristic
            data_obj.strategy = 'sample_div'
            # number of iterations depends on how many layers we have
            divisions = 2
            if N_max < 40:
                divisions = 1
            # the total number of function evaluations for sample division is the first uniform sampling pass and then divisions * that sampling pass
            # so if N_max = 100, and there are 4 divisions, then first pass should be 20, then the following 4 divisions should be 20 each, hence divisions + 1 in computation
            N = int(N_max / (divisions + 1))
            if not file_exists(data_obj):
                sample_div_results = run_iter_resampling(N = N, divisions = divisions, data_obj = data_obj, qm_function = qm_funcs_norm, camera = camera1, bounds = bounds, write = write)
                best_val = round(1 - sample_div_results['best_val'].item(), 6)
                all_res['Sample Division'] = best_val
