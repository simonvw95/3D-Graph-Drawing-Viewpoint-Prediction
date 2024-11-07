import os
import numpy as np
import pickle
import re
from tqdm import tqdm

# this script processes all the raw .txt files into processed .txt files that are averaged over all runs and show a straight progression
# from 10 to 500 function evaluations with intervals of 10

# load in all the graph names that exist (not necessarily the ones that we included in our computed dataset)
graph_names = os.listdir('data')


# funciton that takes as input the directory, textfile name and the maximum number of function evaluations
# processes the txt file into a numpy array of length 500
# for combinations of metrics, divide whatever maximum length (500 e.g.) by the number of metrics (4 metrics -> 500/4=125)
def process_txt(dir, txtfile, N_max = 500):

    # the following four strategies all have parallel runs
    if '-uni_grad-' in txtfile:
        parallels = True
    elif '-grad_desc5-' in txtfile:
        parallels = True
    elif '-sim_anneal5-' in txtfile:
        parallels = True
    elif '-newton_raphson5-' in txtfile:
        parallels = True
    else:
        parallels = False

    # some strategies work with local search steps or a population, meaning that the sequential function evaluations do not always find better results
    # due to exploration. To plot this better, at every 10 results we take the max found so far
    # for strategies that have multiple runs in parallel we simply put them side by side and find the best result of the parallel 10 results
    if parallels:
        # load the file
        f = np.loadtxt(dir + txtfile, dtype = str)
        locations_starts = []
        # find the locations where a new run is started, indicated by start
        for i in range(len(f)):
            if f[i] == 'start':
                locations_starts.append(i)
        # make sure that all are of equal length
        # then put them side by side
        splits = np.split(f, np.argwhere(f == 'start').flatten()[1:])
        new_splits = np.array([])
        n_splits = len(splits)
        segment_size = int(N_max / n_splits)

        for i in range(n_splits):
            # remove the 'start' entry
            splits[i] = splits[i][1:].astype(float)
            size = len(splits[i])
            # if there are 5 runs (5 splits) then each should have N_max / n_splits values
            if size < segment_size:
                # add the max value to the end of the list of numbers
                add = np.repeat(np.nanmax(splits[i]), int(segment_size - size))
                splits[i] = np.append(splits[i], add)
            elif size > segment_size:
                splits[i] = splits[i][0:segment_size]

            new_splits = np.append(new_splits, splits[i])

        # now that the individual runs have been extended or decreased, we put them in parallel
        new_f = []
        for s in range(segment_size):
            for p in range(n_splits):
                new_f.append(new_splits[s + (segment_size * p)])
        f = new_f

    else:
        f = np.loadtxt(dir + txtfile)
        # in cases where, somehow, the max number of function evaluations was above or below N_max (can happen in simulated annealing)
        # either remove anything above N_max, or add the best result to the end to reach N_max length
        size = len(f)
        if size > N_max:
            f = f[:N_max]
        elif size < N_max:
            add = np.repeat(np.max(f), N_max - size)
            f = np.append(f, add)

    # return an array with step length 10 and size 50
    progression = []
    for i in range(10, 510, 10):
        progression.append(np.nanmax(f[0:i]))
    progression = np.array(progression)

    return progression


def process_metric(qm_name, N_max = 500):

    # load in all the names of the files of viewpoint loss progression
    res_dir = 'evaluations/results/' + qm_name + '/'
    result_files = os.listdir(res_dir)

    # strategies that need to be averaged
    strategies_rand = ['-sim_anneal1-', '-sim_anneal5-', '-genetic_alg-', '-pso-', '-grad_desc1-', '-grad_desc5-',
                       '-uni_grad-', '-uni_gradv2-', '-newton_raphson1-', '-newton_raphson5-']
    # strategies that have no randomness but have different behaviors for max N
    strategies_det = ['-uni_sample-', '-sample_div-']

    det_range = list(range(10, 510, 10))
    det_txt = []
    for i in det_range:
        det_txt.append('-' + str(i) + '-i0')

    # loop over all graphs we hae saved
    for graph in tqdm(graph_names):

        graph_computed = False
        strat_results = {key: [] for key in strategies_rand}

        strat_det_results = {key: {subkey: 0 for subkey in det_txt} for key in strategies_det}
        # get all the files related to this graph
        for res in result_files:
            if graph in res:
                graph_computed = True
                # get all the files related to each specific strat and dump the processed txt file into a dictionary
                for strat in strategies_rand:
                    if strat in res:
                        processed_txt = process_txt(res_dir, res, N_max = N_max)
                        if len(processed_txt) != 0:
                            strat_results[strat].append(processed_txt)
                # process the results of uniform sampling and sample division by getting the best result for every run of 10-500 func evaluations
                for strat in strategies_det:
                    if strat in res:
                        for it in strat_det_results[strat]:
                            if it in res:
                                strat_det_results[strat][it] = np.max(np.loadtxt(res_dir + res))

        if graph_computed:
            # now we can average over the multiple runs
            for strat in strategies_rand:

                means_i = np.mean(np.array(strat_results[strat]), axis = 0)

                if not os.path.isdir('evaluations/processed_results/' + qm_name):
                    os.makedirs('evaluations/processed_results/' + qm_name + '/')

                full_name = qm_name + '-' + graph + '-' + strat
                f = open('evaluations/processed_results/' + qm_name + '/' + full_name + '.txt', 'w')
                for val in means_i:
                    f.write(str(val) + '\n')
                f.close()

            for strat in strategies_det:

                full_array = list(strat_det_results[strat].values())

                full_name = qm_name + '-' + graph + '-' + strat
                f = open('evaluations/processed_results/' + qm_name + '/' + full_name + '.txt', 'w')
                for val in full_array:
                    f.write(str(val) + '\n')
                f.close()

    processed_result_files = os.listdir('evaluations/processed_results/' + qm_name + '/')
    strat_full_names = ['Uniform Sample', 'Iterative Resampling', 'Simulated Annealing (1)', 'Simulated Annealing (5)', 'Differential Evolution', 'Particle Swarm Optimization', 'Gradient Descent (1)', 'Gradient Descent (5)', 'Uniform Gradient Descent V1', 'Uniform Gradient Descent V2', 'Newton-Raphson (1)', 'Newton-Raphson (5)']

    name_dict = dict(zip(strategies_det + strategies_rand, strat_full_names))

    # put all grpah results in a single dictionary, where keys are strats, and values are graph dictionaries with nested arrays

    strat_results = {key: {} for key in strategies_det + strategies_rand}
    for f in tqdm(processed_result_files):
        for strat in strat_results:
            if strat in f:
                graphname = re.search(r'' + qm_name + '-(.*?)' + strat, f).group(1)[:-1]
                strat_results[strat][graphname] = np.loadtxt("evaluations/processed_results/" + qm_name + "/" + f)

    # createa  .pkl file with the results averaged over all graphs
    final_dict = {}
    for strat in strat_results:
        final_dict[name_dict[strat]] = np.mean(np.array(list(strat_results[strat].values())), axis = 0)

    with open('evaluations/processed_results/' + qm_name + '.pkl', 'wb') as handle:
        pickle.dump(final_dict, handle)

    # create a .pkl file with results not averaged over all graphs but having each graph separately in the dict
    final_dict = {}
    for strat in strat_results:
        final_dict[name_dict[strat]] = strat_results[strat]

    with open('evaluations/processed_results/' + qm_name + 'nonmean.pkl', 'wb') as handle:
        pickle.dump(final_dict, handle)


# process_metric(qm_name = 'norm_stress_torch_pairs1', N_max = 500)
# process_metric(qm_name = 'edge_lengths_sd_torch1', N_max = 500)
# process_metric(qm_name = 'node_occlusion1', N_max = 500)
# process_metric(qm_name = 'node_edge_occlusion1', N_max = 500)
# process_metric(qm_name = 'crossings_number1', N_max = 500)
# process_metric(qm_name = 'norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1', N_max = 125)
