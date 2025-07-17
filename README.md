# Last-Mile Routing with GNN and Pointer Network

This repository contains the code to perfom training and inference for the last-mile routing problem with the Amazon Last Mile Routing Challenge 2021 Dataset. The model includes a GNN encoder and a Pointer Network decoder. The code available includes two approaches: a general train with the whole data and a zone-based train with an instance of the model trained per zone. The main files are inside **PyTorch_GNN** folder.

## Models

The encoder is defined in **GNNEncoder.py**, while the decoder is included in **PointerDecoder.py**.


## Train

**PyTorch_GNN.py** file contains the general training. Hyperparameters are defined in **\_\_main\_\_** and can be changed for training. It tries to read a file with the generated graphs. If it does not find them, it generates again the graphs and stores them in a file.

**Zone_training.py** includes the code to train the models of each zone. It must be launched specifyng the number of the cluster to train (e. g. _python Zone\_training.py 0_). **zone_train.sh** automates the process with a loop that launches the training of all specified clusters.


## Test

**PyTorch_Test.ipynb** contains the code for inferences in test dataset and to measure the error and the inference times.

**Zone_inference.py** performs the zone-based inference for test routes. It stores the results in JSON files. Those files are read in **Zone_inference_results.ipynb** to extract statistics about errors and inference times.


## Auxiliar files

**read_best_checkpoint.ipynb** contains the code for checking the best validation time of a trained model and the epoch in which it has been achieved. Also, from the checkpoint the weights of the best model can be obtained.

**plot_graph.ipynb** includes code to plot a route with graph structure.

**utils.py** contains auxiliar code for data preparation and visualization.

**zone_analysis.ipynb** contains code to check statistics about the generated zones (e.g. the number of clusters visited by each route).


## Folders

**results/** contains the plot with train and validation lengths per epoch and the best checkpoint of the general training.

**zone_dfs/** contains the subroutes of each cluster in parquet format.

**zone_results/** contains the models generated for each zone and the predicted times in inference process.

**zone_times/** contains JSON files with the training and inference times per cluster.