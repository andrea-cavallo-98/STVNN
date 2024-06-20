# STVNN

This repository contains the code for the Spatio-Temporal coVariance Neural Network (STVNN). 

## Description

STVNN is a model for multivariate time series processing and forecasting. At different time snapshots, it performs convolutions using the sample covariance matrix as graph and it aggregates information over time to model temporal dependencies. A single layer produces the output

$$ \textbf{z}_t = \sum\_{t'=0}^{T-1}\sum\_{k=0}^K h\_{kt'} \mathbf{\hat{C}}_t^k \mathbf{x}\_{t-t'}.$$

To account for streaming data and distribution shifts, STVNN is updated online. It is provably stable to covariance estimation uncertainties and it adapts to changes in the data distribution.

![STVF](./figures/STVNN.svg)  

## Repository structure
- `baselines`: contains the code to run the baselines LSTM, TPCA, VNN with temporal features and VNN-LSTM
- `Data`: real datasets
- `graphML.py`, `graphTools.py`: generic functions for graph processing
- `layers.py`, `models.py`: implementation of main model and baselines
- `main_conv.py`: code to run STVNN experiments, see below
- `utils.py`: generic functions

## Usage
The following snippet is an example of an experiment using STVNN.
```
python main_conv.py --pred_step 1 --T 5 --optimizer Adam --lr 0.001 --nEpochs 40 --gamma 0.01 --dimNodeSignals 1,128,64 --filter_taps 2 --dimLayersMLP 64,32,1 --dset Molene
```

The experiments with baselines, instead, can be run using the following commands.
```
cd baselines
python main_vnn.py --dset exchange_rate
```

Parameters:
- `pred_step`: how many steps in the future to predict
- `T`: size of temporal window
- `optimizer`: `Adam` or `SGD`
- `lr`: learning rate
- `nEpochs`: how many epochs for training
- `stationary`: whether to use the stationary or non-stationary covariance update (default `False`)
- `gamma`: parameter for non-stationary covariance update (should be a real value between 0 and 1)
- `dimNodeSignals`: size of the filterbanks as a csv string
- `filter_taps`: order of the graph filters (corresponds to $K + 1$)
- `dimLayersMLP`: size of the layers of MLP for the final task as a csv string
- `batchSize`: batch size, set to 1 for the online setting
- `dset`: `Molene`, `exchange_rate` or `NOA`.

## Requirements

Requirements can be installed with

```
pip install -r requirements.txt
```

## Citation
If you find this code useful, please cite
```
@InProceedings{cavallo2024stvnn,
author="Cavallo, Andrea and Sabbaqi, Mohammad and Isufi, Elvin",
title="Spatiotemporal Covariance Neural Networks",
booktitle="Machine Learning and Knowledge Discovery in Databases: Research Track",
year="2024",
}
```

## Notes

Part of the code is borrowed from [https://github.com/alelab-upenn/graph-neural-networks](https://github.com/alelab-upenn/graph-neural-networks).
