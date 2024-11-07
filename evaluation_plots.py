import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import torch
import copy
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from matplotlib import gridspec

warnings.simplefilter(action='ignore', category=FutureWarning)

# change to latex font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# default values for most plots
width = 1800
height = 1200
dpi = 96
xlabel_font = 40
ylabel_font = 40
xlabel_pad = 25
ylabel_pad = 25
major_tick = 30
minor_tick = 28

# colors and markers for the different strategies
key_marker_color = {'Ground Truth' : ['', 'gray'], 'Uniform Sample' : ['s', 'black'], 'Iterative Resampling' : ['X', 'black'], 'Simulated Annealing (1)' : ['s', 'blue'], 'Simulated Annealing (5)' : ['X', 'blue'], 'Differential Evolution' : ['P', 'blue'], 'Particle Swarm Optimization' : ['v', 'blue'], 'Gradient Descent (1)' : ['s', 'red'], 'Gradient Descent (5)' : ['X', 'red'], 'Uniform Gradient Descent V1' : ['P', 'red'], 'Uniform Gradient Descent V2' : ['v', 'red'], 'Newton-Raphson (1)' : ['s', '#4daf4a'], 'Newton-Raphson (5)' : ['X', '#4daf4a'], 'DeepGD' : ['P', '#4daf4a']}


# function for creating jitter plots of the distributions of viewpoint optimization values for all graphs
def create_jitter_plots(N, qm_name, deepgd = False, ground_truth_incl = True):

    # open the results of the quality metric
    with open('evaluations/processed_results/' + qm_name + 'nonmean.pkl', 'rb') as handle:
        data_dict = pickle.load(handle)

    # load the ground truth
    with open('evaluations/ground_truth-' + qm_name + '.pkl', 'rb') as handle:
        e = pickle.load(handle)

    # scale by subtracting the ground truth to see the difference
    for key in data_dict:
        for graph in e:
            if graph in data_dict[key]:
                # simply subtract the ground truth from the value
                data_dict[key][graph] = data_dict[key][graph] - e[graph]['Ground Truth']['best_loss']

    # we don't need the graph labels anymore in jitter plots so put all values into nested array
    for key in data_dict:
        data_dict[key] = np.array(list(data_dict[key].values()))

    # only interested in the values at a certain number of function evaluations (N)
    # but the dictionary already has values from 10-500 with intervals of 10 so
    # divide the interested N by 10 to get the requested range from the .pkl file
    new_data_dict = {}
    for key in data_dict:
        new_data_dict[key] = data_dict[key][:, int(N / 10)]

    data_dict = new_data_dict

    # if we want to include the ground truth in the jitter plot
    if ground_truth_incl:
        # add ground truth data to the dictionary and make it the first key in the dictionary for visualization purposes
        data_dict['Ground Truth'] = np.array([])

        for key in e:
            data_dict['Ground Truth'] = np.append(data_dict['Ground Truth'], (e[key]['Ground Truth']['best_loss']) - (e[key]['Ground Truth']['best_loss']))

    # if we want to include deepgd results in the jitter plot (only for stress available)
    if deepgd:
        # load deepgd results
        key_order = ['Ground Truth'] + list(data_dict.keys()) + ['DeepGD']
        with open('evaluations/deepgd_results_unclamped.pkl', 'rb') as handle:
            dpgd_norm = pickle.load(handle)

        data_dict['DeepGD'] = np.array([])
        for key in dpgd_norm:
            data_dict['DeepGD'] = np.append(data_dict['DeepGD'], dpgd_norm[key]['DeepGD (norm)']['loss'] - (e[key]['Ground Truth']['best_loss']))
    else:
        key_order = ['Ground Truth'] + list(data_dict.keys())

    if not ground_truth_incl:
        key_order = list(data_dict.keys())

    # change the key order of the dictionary,  useful for visualizing later
    new_data_dict = {}
    for key in key_order:
        new_data_dict[key] = data_dict[key]

    data_dict = new_data_dict

    # a list of all the strategies to feed to the jitter plot
    strat_list_tot = []
    strat_keys = list(data_dict.keys())
    for strat in strat_keys:
        strat_list_tot += [[strat] * len(data_dict[strat])]
    strat_list_tot = np.array(strat_list_tot).flatten()

    # a list of all the values corresponding to the strategy list to feed to the stripplot
    strat_res = np.array([])
    for key in strat_keys:
        strat_res = np.append(strat_res, data_dict[key])

    full_val_list = strat_res.flatten()

    # color list
    color_list = []
    for key in strat_keys:
        color_list.append(key_marker_color[key][1])

    # ugly way of doing this but for now this will do
    if qm_name == 'norm_stress_torch_pairs1':
        qm_write = r'\texttt{ST}'
    elif qm_name == 'edge_lengths_sd_torch1':
        qm_write = r'\texttt{ELD}'
    elif qm_name == 'node_occlusion1':
        qm_write = r'\texttt{NN}'
    elif qm_name == 'node_edge_occlusion1':
        qm_write = r'\texttt{NE}'
    elif qm_name == 'crossings_number1':
        qm_write = r'\texttt{CN}'
    elif qm_name == 'norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1':
        qm_write = r'\texttt{LC}'

    # create the matpltolib figure
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()

    # crate the dataframe
    end_data = pd.DataFrame({'Strategies': strat_list_tot, qm_write: full_val_list})
    ts = sns.stripplot(data=end_data, x='Strategies', y=qm_write, size=10, ax=ax, jitter=0.25, hue = 'Strategies', palette = color_list, edgecolor = 'gray', linewidth = 0.5)

    mean_width = 0.45

    # plot the mean line
    for tick, text in zip(ax.get_xticks(), strat_keys):
        curr_name = text

        mean_val = end_data[end_data['Strategies'] == curr_name][qm_write].mean()
        # plot horizontal lines across the column
        ax.annotate(str(round(mean_val, 6)),
                    xy=(tick, mean_val), xycoords='data',
                    xytext=(-8, 2), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='bottom')
        ax.plot([tick - mean_width / 2, tick + mean_width / 2], [mean_val, mean_val], lw=3, color='black')

    # use a boxplot to get the limits on the y axis
    bx = sns.boxplot(showmeans=False,
                meanline=False,
                meanprops={'visible' : False, "marker": "+", "markeredgecolor": "red", "markersize": "15"},
                medianprops={'visible': False, 'linewidth' : '2.5', },
                whiskerprops={'visible': False},
                zorder=10,
                x="Strategies",
                y=qm_write,
                data=end_data,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ts)

    ylims = bx.get_ylim()
    ymax = ylims[1]
    # cut off all outliers in case they are worse than the worst of the uniform sample technique
    ymin = np.min(data_dict['Uniform Sample'])
    ax.tick_params(axis='both', which='major', labelsize=major_tick)
    ax.tick_params(axis='both', which='minor', labelsize=minor_tick)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax.set(ylim=(ymin, ymax))
    ax.legend().set_visible(False)
    plt.xlabel('Strategy', fontsize=xlabel_font, labelpad = xlabel_pad)
    plt.ylabel(qm_write + ': Viewpoint - Ground Truth', fontsize=ylabel_font, labelpad = ylabel_pad)
    save_name = 'evaluations/' + qm_name + '-jitter_plots-' + str(N) + 'scal.pdf'
    plt.savefig(save_name, bbox_inches= "tight")
    plt.clf()
    plt.close('all')


create_jitter_plots(N = 300, qm_name = 'norm_stress_torch_pairs1', deepgd = True)
# create_jitter_plots(N = 300, qm_name = 'edge_lengths_sd_torch1')
# create_jitter_plots(N = 300, qm_name = 'node_occlusion1')
# create_jitter_plots(N = 300, qm_name = 'node_edge_occlusion1')
# create_jitter_plots(N = 300, qm_name = 'crossings_number1')
# create_jitter_plots(N = 300, qm_name = 'norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1', scale = True, ground_truth_incl = False)

# create_jitter_plots(N = 40, qm_name = 'norm_stress_torch_pairs1', deepgd = True)
# create_jitter_plots(N = 40, qm_name = 'edge_lengths_sd_torch1')
# create_jitter_plots(N = 40, qm_name = 'node_occlusion1')
# create_jitter_plots(N = 40, qm_name = 'node_edge_occlusion1')
# create_jitter_plots(N = 40, qm_name = 'crossings_number1')
# create_jitter_plots(N = 40, qm_name = 'norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1', ground_truth_incl = False)
# print('done making jitter plots')


##############################################################################################################################


# creating strategies over iterations
def individ_plot(qm_name, qm, ground_truth_incl = True):

    with open('evaluations/processed_results/' + qm_name + '.pkl', 'rb') as handle:
        data_dict = pickle.load(handle)

    # load the ground truth
    with open('evaluations/ground_truth-' + qm_name + '.pkl', 'rb') as handle:
        e = pickle.load(handle)

    if qm == r'\texttt{ST}':
        # load deepgd results
        with open('evaluations/deepgd_results_unclamped.pkl', 'rb') as handle:
            dpgd_norm = pickle.load(handle)

    # add groudn truth and deepgd results
    ground_truth_array = []
    if qm == r'\texttt{ST}':
        deepgd_array = []
    for key in e:
        ground_truth_array.append(e[key]['Ground Truth']['best_loss'])
        if qm == r'\texttt{ST}':
            deepgd_array.append(dpgd_norm[key]['DeepGD (norm)']['loss'])

    ground_truth = np.repeat(np.mean(ground_truth_array), len(data_dict['Uniform Sample']))

    if ground_truth_incl:
        data_dict['Ground Truth'] = ground_truth

    if qm == r'\texttt{ST}':
        deepgd = np.repeat(np.mean(deepgd_array), len(data_dict['Uniform Sample']))
        data_dict['DeepGD'] = deepgd

    # add ground truth data to the dictionary and make it the first key in the dictionary for visualization purposes
    if ground_truth_incl:
        key_order = ['Ground Truth'] + list(data_dict.keys())
    else:
        key_order = list(data_dict.keys())

    if qm == r'\texttt{ST}':
        key_order = ['Ground Truth'] + list(data_dict.keys()) + ['DeepGD']

    # change key order
    new_data_dict = {}
    for key in key_order:
        new_data_dict[key] = data_dict[key]

    # create the plot including every strategy while subtracting the ground truth
    x = np.array(range(10, 510, 10))
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=major_tick)
    ax.tick_params(axis='both', which='minor', labelsize=minor_tick)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    perc_dict = copy.deepcopy(new_data_dict)
    strats_perc = list(perc_dict.keys())
    for strat in strats_perc:
        # closeness to ground truth by subtraction
        perc_dict[strat] = perc_dict[strat] - ground_truth

    markers = []
    colors = []
    for strat in strats_perc:
        markers.append(key_marker_color[strat][0])
        colors.append(key_marker_color[strat][1])
    plot_name = 'evaluations/' + qm_name + 'scal.pdf'

    for i in range(len(strats_perc)):
        pl_i = plt.plot(x, perc_dict[strats_perc[i]], label = strats_perc[i], linestyle = '-', marker = markers[i], color = colors[i], markersize = 8)

    plt.xlabel('Number of Function Evaluations', fontsize=xlabel_font, labelpad = xlabel_pad)
    plt.ylabel(qm + ': Viewpoint - Ground Truth', fontsize=ylabel_font, labelpad = ylabel_pad)
    ax.legend(loc = 'lower right', prop={'size': 20})

    plt.savefig(plot_name, bbox_inches= "tight")
    plt.clf()
    plt.close('all')

    # create the plot including every strategy minus the ground truth but exclude Gradient Descent 1 and DeepGD

    x = np.array(range(10, 510, 10))
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=major_tick)
    ax.tick_params(axis='both', which='minor', labelsize=minor_tick)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    perc_dict = copy.deepcopy(new_data_dict)
    strats_perc = list(perc_dict.keys())

    strats_perc.remove('Gradient Descent (1)')
    if 'DeepGD' in strats_perc:
        strats_perc.remove('DeepGD')

    for strat in strats_perc:
        # closeness to ground truth by subtraction
        perc_dict[strat] = perc_dict[strat] - ground_truth

    markers = []
    colors = []
    for strat in strats_perc:
        markers.append(key_marker_color[strat][0])
        colors.append(key_marker_color[strat][1])
    plot_name = 'evaluations/' + qm_name + 'scal_excl.pdf'

    for i in range(len(strats_perc)):
        pl_i = plt.plot(x, perc_dict[strats_perc[i]], label = strats_perc[i], linestyle = '-', marker = markers[i], color = colors[i], markersize = 8)

    plt.xlabel('Number of Function Evaluations', fontsize=xlabel_font, labelpad = xlabel_pad)
    plt.ylabel(qm + ': Viewpoint - Ground Truth', fontsize=ylabel_font, labelpad = ylabel_pad)
    ax.legend(loc = 'lower right', prop={'size': 20})

    plt.savefig(plot_name, bbox_inches= "tight")
    plt.clf()
    plt.close('all')

    print('done')


# individ_plot('norm_stress_torch_pairs1', r'\texttt{ST}', ground_truth_incl = True)
# individ_plot('edge_lengths_sd_torch1', r'\texttt{ELD}', ground_truth_incl = True)
# individ_plot('node_occlusion1', r'\texttt{NN}', ground_truth_incl = True)
# individ_plot('node_edge_occlusion1', r'\texttt{NE}', ground_truth_incl = True)
# individ_plot('crossings_number1', r'\texttt{CN}', ground_truth_incl = True)
# individ_plot('norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1', r'\texttt{LC}', ground_truth_incl = False)


###################################################################################################################################


# function to create a plot that puts the results of all graphs of all metrics together
def merged_strats(N, grad_desc_incl = True):

    mass_data_dict = {}

    # loop over all quality metrics
    for qms in ['norm_stress_torch_pairs1', 'edge_lengths_sd_torch1', 'node_occlusion1', 'node_edge_occlusion1', 'crossings_number1']:
        qm_name = qms

        # load the qm results
        with open('evaluations/processed_results/' + qm_name + '.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)

        # load the ground truth
        with open('evaluations/ground_truth-' + qm_name + '.pkl', 'rb') as handle:
            e = pickle.load(handle)

        # only interested in the values at a certain number of function evaluations (N)
        new_data_dict = {}
        for key in data_dict:
            new_data_dict[key] = data_dict[key][int(N / 10)]

        data_dict = new_data_dict

        # add ground truth data to the dictionary and make it the first key in the dictionary for visualization purposes
        data_dict['Ground Truth'] = np.array([])
        key_order = ['Ground Truth'] + list(data_dict.keys())
        best_loss = []
        for key in e:
            data_dict['Ground Truth'] = np.append(data_dict['Ground Truth'], (e[key]['Ground Truth']['best_loss']))
            best_loss.append(e[key]['Ground Truth']['best_loss'])
        data_dict['Ground Truth'] = np.mean(data_dict['Ground Truth'])

        # scale via ground truth average
        for key in data_dict:
            data_dict[key] = data_dict[key] - np.mean(best_loss)

        # change the order of the keys
        new_data_dict = {}
        for key in key_order:
            new_data_dict[key] = data_dict[key]

        data_dict = new_data_dict

        if not grad_desc_incl:
            data_dict.pop('Gradient Descent (1)', None)

        if qm_name == 'norm_stress_torch_pairs1':
            qm_write = r'\texttt{ST}'
        elif qm_name == 'edge_lengths_sd_torch1':
            qm_write = r'\texttt{ELD}'
        elif qm_name == 'node_occlusion1':
            qm_write = r'\texttt{NN}'
        elif qm_name == 'node_edge_occlusion1':
            qm_write = r'\texttt{NE}'
        elif qm_name == 'crossings_number1':
            qm_write = r'\texttt{CN}'
        elif qm_name == 'norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1':
            qm_write = r'\texttt{LC}'

        mass_data_dict[qm_write] = data_dict

    # create the plot while including every strategy and the ground truth
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=major_tick)
    ax.tick_params(axis='both', which='minor', labelsize=minor_tick)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    strats = mass_data_dict[list(mass_data_dict.keys())[0]]

    markers = []
    colors = []
    for strat in strats:
        markers.append(key_marker_color[strat][0])
        colors.append(key_marker_color[strat][1])

    if grad_desc_incl:
        plot_name = 'evaluations/all_metrics_strat-N' + str(N) + '.pdf'
    else:
        plot_name = 'evaluations/all_metrics_strat_excl-N' + str(N) + '.pdf'

    for strat in strats:
        x_axis = list(mass_data_dict.keys())
        vals = []
        for qm in x_axis:
            vals.append(mass_data_dict[qm][strat])
        pl_i = plt.plot(x_axis, vals, label = strat, linestyle = '-', marker = key_marker_color[strat][0], color = key_marker_color[strat][1], markersize = 8)

    plt.xlabel('Quality Metrics', fontsize= xlabel_font, labelpad = xlabel_pad)
    plt.ylabel('Quality Metric: Viewpoint - Ground Truth', fontsize=ylabel_font, labelpad = ylabel_pad)
    ax.legend(loc = 'lower left', prop={'size': 20})

    plt.savefig(plot_name, bbox_inches= "tight")
    plt.clf()
    plt.close('all')


# merged_strats(N = 40, grad_desc_incl = True)
# merged_strats(N = 40, grad_desc_incl = False)
# merged_strats(N = 300, grad_desc_incl = True)
# merged_strats(N = 300, grad_desc_incl = False)
# print('done with all strats together')


###################################################################################################################################################


# function for creating drawings of graphs for which we manually got the viewpoints (manual_comparison.py)
def create_drawings(qm_name):
    N_maxs = [40, 300]

    print('Doing metric: ' + qm_name)
    for N_max in N_maxs:
        print('Doing N_max: ' + str(N_max))
        all_strats = list(key_marker_color.keys())

        with open('evaluations/drawings/' + qm_name + '-' + str(N_max) + '.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)

        new_data_dict = {}
        for dataset in data_dict:
            new_data_dict[dataset] = {}
        # redo ordering based on all_strats keys ordering
        for dataset in data_dict:
            for subkey in all_strats:
                if subkey in data_dict[dataset]:
                    new_data_dict[dataset][subkey] = data_dict[dataset][subkey]

        if qm_name == 'norm_stress_torch_pairs1':
            with open('evaluations/deepgd_results_manual5.pkl', 'rb') as handle:
                dpgd_res = pickle.load(handle)

            for dataset in new_data_dict:
                new_data_dict[dataset]['DeepGD'] = dpgd_res[dataset]['DeepGD (norm)']

        # visualize graphs
        print('visualizing the graphs')
        spec_dir = 'evaluations/drawings/' + qm_name + '-' + str(N_max) + '/'
        device = 'cpu'
        camera1 = PerspectiveCameras(device = device, focal_length = torch.tensor([1]).float().to(device))
        # get the 2d coordinates
        for dataset in new_data_dict:
            print('Doing dataset: ' + dataset)
            for subkey in new_data_dict[dataset]:
                elevation, azimuth = new_data_dict[dataset][subkey]['vwp'][0:2]

                if type(elevation) != torch.Tensor:
                    elevation = torch.tensor(elevation)
                if type(azimuth) != torch.Tensor:
                    azimuth = torch.tensor(azimuth)

                elevation = elevation.float()
                azimuth = azimuth.float()

                # get the transformation matrix
                R, T = look_at_view_transform(1, elevation, azimuth, camera1.device)
                transform_matrix = camera1.get_full_projection_transform(R = R.float().to(camera1.device), T = T.float().to(camera1.device)).get_matrix()[0]

                # apply the transformation matrix to the 3d coords to get 2d coords (view)
                layout_file_3d = "layouts/" + dataset + "-FA2-3d.csv"
                coords_3d = torch.tensor(pd.read_csv(layout_file_3d, sep=';').to_numpy()).float()

                df = pd.read_csv("data/" + dataset + "/" + dataset + "-src.csv", sep=';', header=0)
                G = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
                G = nx.convert_node_labels_to_integers(G)
                # some graphs sometimes have selfloops
                G.remove_edges_from(nx.selfloop_edges(G))

                n = coords_3d.shape[0]
                projection = torch.ones((n, 4))

                projection[0:n, :3] = coords_3d
                view = torch.matmul(projection.to(camera1.device), transform_matrix.float())[:, 0:2]

                # Scale exactly to the range (0, 1)
                view = view - torch.min(view, axis=0)[0]
                coords_2d = view / torch.max(view)

                coords_dict = {}
                for i in range(len(coords_2d)):
                    coords_dict[i] = [coords_2d[i][0].item(), coords_2d[i][1].item()]

                plot_name = spec_dir + qm_name + '-' + dataset + '-' + subkey + '.png'
                width = 1800
                height = 1800
                dpi = 96
                plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                nx.draw(G, with_labels = False, pos = coords_dict, node_size = 20, width = 3.0, edge_color = 'black')
                plt.savefig(plot_name, bbox_inches="tight", pad_inches = -1)
                plt.clf()
                plt.close('all')


# create_drawings('norm_stress_torch_pairs1')
# create_drawings('edge_lengths_sd_torch1')
# create_drawings('node_occlusion1')
# create_drawings('node_edge_occlusion1')
# create_drawings('crossings_number1')
# create_drawings('norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1')


###################################################################################################################################################


# function for putting all the existing drawings created by the script above in a single big pdf
def create_drawings_large(qm_name):

    graph_names_normal = ['sierpinski3d', 'GD96_c', 'L', 'can_96', 'dwt_1005']
    graph_names_cn = ['GD96_c', 'mesh1em6', 'gridaug', 'grafo6975', 'grafo10230']
    N_maxs = [40, 300]

    for N_max in N_maxs:
        all_strats = list(key_marker_color.keys())

        with open('evaluations/drawings/' + qm_name + '-' + str(N_max) + '.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)

        graph_results = {key: {} for key in data_dict}
        for dataset in data_dict:
            for subkey in data_dict[dataset]:
                if subkey != 'Ground Truth':
                    graph_results[dataset][subkey] = data_dict[dataset][subkey]['loss']

        winners = {}
        for dataset in graph_results:
            winners[dataset] = sorted(graph_results[dataset], key=graph_results[dataset].get, reverse=True)[0:3]

        if qm_name != 'norm_stress_torch_pairs1':
            all_strats.remove('DeepGD')
        if 'crossings_number' in qm_name:
            graph_names = graph_names_cn
        else:
            graph_names = graph_names_normal

        nrow = len(all_strats)
        ncol = len(graph_names)

        fig = plt.figure(figsize=(ncol, nrow))

        gs = gridspec.GridSpec(nrow, ncol, width_ratios= list(np.ones(ncol)), height_ratios= list(np.ones(nrow)),
                 wspace=0.05, hspace=0.0, top=0.99, bottom=0.01, left=0.15, right=0.99, figure = fig)

        for j in range(len(all_strats)):

            curr_strat = all_strats[j]
            for i in range(len(graph_names)):
                curr_graph = graph_names[i]

                place = -1
                for h in range(len(winners[curr_graph])):
                    if curr_strat == winners[curr_graph][h]:
                        place = h

                # gold border
                if place == 0:
                    box_color = 'red'
                elif place == 1:
                    box_color = 'blue'
                elif place == 2:
                    box_color = 'yellow'

                name = 'evaluations/drawings/' + qm_name + '-' + str(N_max) + '/' + qm_name + '-' + curr_graph + '-' + curr_strat + '.png'

                img = plt.imread(name)

                ax = plt.subplot(gs[j, i])
                if i == 0:
                    ax.set_ylabel(curr_strat, rotation=45, fontsize=4, labelpad=45)
                ax.imshow(img)
                ax.tick_params(left=False, right=False, labelleft=False,
                                      labelbottom=False, bottom=False, top=False, labeltop=False)
                if place < 0:
                    ax.set_frame_on(False)
                else:
                    ax.set_frame_on(True)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(box_color)
                if j == 0:
                    ax.set_xlabel(curr_graph, fontsize=4, rotation = 0, labelpad = 5)
                    ax.xaxis.set_label_position('top')

                if i == 0:
                    ax.set_ylabel(curr_strat, rotation=45, fontsize=4, labelpad=25)

        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)

        fig.savefig('evaluations/drawings/' + qm_name + '-' + str(N_max) + '.pdf', dpi = 1000)
        fig.clf()
        plt.close('all')
        print('done')


# create_drawings_large('norm_stress_torch_pairs1')
# create_drawings_large('edge_lengths_sd_torch1')
# create_drawings_large('node_occlusion1')
# create_drawings_large('node_edge_occlusion1')
# create_drawings_large('crossings_number1')
# create_drawings_large('norm_stress_torch_pairs1edge_lengths_sd_torch0.8node_occlusion0.9crossings_number1')
