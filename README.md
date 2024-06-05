# STVNN

This repository contains the code for the Spatio-Temporal coVariance Neural Network (STVNN). 

## Description

STVNN is a model for multivariate time series processing and forecasting. At different time snapshots, in performs convolutions using the sample covariance matrix as graph and it aggregates information over time to model temporal dependencies. A single layer produces the output

$$\textbf{z}_t = \sum_{t'=0}^{T-1}\sum_{k=0}^Kh_{kt'}\textbf{\hat{C}}_t^k \textbf{x_{t-t'}}$$.

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
The supported datasets are `Molene`, `exchange_rate` and `NOA`.

## Requirements

All requirements can be installed with

```
pip install requirements.txt
```

## Notes

Part of the code is borrowed from [https://github.com/alelab-upenn/graph-neural-networks](https://github.com/alelab-upenn/graph-neural-networks).