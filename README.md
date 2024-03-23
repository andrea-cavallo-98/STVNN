# STVNN

This repository contains the code for the Spatio-Temporal coVariance Neural Network (STVNN). 

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

## Notes

Part of the code is borrowed from [https://github.com/alelab-upenn/graph-neural-networks](https://github.com/alelab-upenn/graph-neural-networks).