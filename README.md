# Viewpoint Optimization in 3D Graph Drawing

# Running instructions

Python version 3.10.4 \
Install packages with: 'pip install -r requirements.txt' 

### Minimal example
With a single git clone (make sure you use git lfs to properly clone the csv files or dowload the csv files manually), the scripts should be usable for all graphs given in the /data/ folder. 

1. Run **auto_comparison.py** to automatically apply every optimization strategy to every graph in the dataset and record the results in .txt files in /evaluations/results. Variables inside the script can be adjusted to reduce or increase the number of graphs in the dataset, the maximum number of function evaluations, whether you want to write the results to .txt files, and more.

2. Run **manual_comparison.py** to automatically apply every optimization strategy to a selection of graphs and save the best viewpoints, and their metric value, to a .pkl file which can then be opened and used to draw the graph drawings in another script.

3. Run **ground_truth.py** to acquire a .pkl file of our definition of the 'ground truth' for a particular metric defined in the script.

4. Run **process_results.py** and specify the metric to be processed in the script. This averages results over multiple runs, processes the .txt files to be a fixed length, puts parallel runs in parallel, and produces averaged .txt files and .pkl files. 

5. Run **evaluation_plots.py**. Uncomment lines for creating specific plots with specific metrics.

6. It is not necessary to run **neural_aesthete.py**. The script contains the MLP class and a function to train a new MLP for predicting crossings from scratch.  

### Personal dataset/graphs

Add source files of your graphs to the /data/ directory by adding a directory with the graphsname and name the graph file as such "graphname-src.csv". E.g. /data/3elt/3elt-src.csv . The graph file should be a .csv containing the edgelist. For the precise format (including delimiter) view one of the example graphs. Repeat same process but also for the shortest path distance matrix in e.g. /data/3elt/3elt-gtds.csv


### Troubleshooting
When cloning the repository, ensure you are using git lfs to clone the project. When git lfs is not used the .csv files in the data directory will only point towards the uploaded csv files.






