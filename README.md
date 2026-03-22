# Project: Trajectory Prediction on Graphs

This project aims to develop a trajectory prediction model using graph-based techniques. By leveraging the inherent structure of graphs, we can capture the relationships between different nodes (street intersections) and make accurate predictions about the future movements based on a short history.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)

## Introduction
In many real-world scenarios, such as autonomous driving and traffic flow prediction, accurately predicting the future trajectory of objects is crucial. This project explores the use of (equivariant) graph-based methods to improve trajectory prediction accuracy by considering the underlying graph structure.

## Installation
To use this project, follow these steps:
1. Clone the repository: `git clone https://gitlab.lrz.de/ml-lab/ss24/trajectory-prediction-on-graphs`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
To run the trajectory prediction model, execute the following command:
```
python train.py --dataset <DATASET> --model <MODEL>
```
where `<DATASET>` is the name of the dataset to use (`pneuma`, `geolife` or `tdrive`) and `<MODEL>` is the name of the model to train (e.g., `gat`, `egnn`, etc.). See the `train.py` script for more options and hyperparameters.

## Data
The datasets used in this project consists of historical trajectory data of taxis in Beijing (`tdrive` and `geolife`) and Athen (`pneuma`). We performed two splits per dataset into training/validation and test set:
- Random Split: Randomly split the data into training/validation and test set.
- Spatial Split: Split the data based on the spatial location of the trajectories, i.e. the training/validation set contains trajectories from one region and the test set from another region.

## Models and Baselines
We compare more advanced models with simple baselines to evaluate the performance of our models. The following models are implemented in this project:
- Random Model
- Multi-Layer Perceptron (MLP)
- Markov Chain (MC)
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- Simplicial Complex Network (SCoNe)
- Equivariant Graph Neural Network (EGNN)

## Results
We evaluated all models on the datasets and documented the results in the [Wiki page](https://collab.dvb.bayern/display/TUMmllab/Project+4+-+Trajectory+prediction+on+Graphs).
