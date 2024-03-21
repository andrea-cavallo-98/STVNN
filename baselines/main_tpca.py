import sys 
sys.path.append('../')

import pandas as pd
import numpy as np
import torch 
from torch import nn, optim
from models import Regressor
from copy import deepcopy
from utils import *
from tqdm import tqdm

online_test = True
pca = True

args = parse_args()

dset = args.dset

m = args.m
pred_step = args.pred_step # how many steps in the future we want to predict
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

T = args.T
gamma = args.gamma
dimNodeSignals = args.dimNodeSignals
L = len(dimNodeSignals) - 1
nFilterTaps = [args.filter_taps] * L
dimLayersMLP = [1]
online = args.online
write = args.out_file is not None
h_size = args.h_size

Xfold, nTotal, y, Yscaler = load_and_normalize_data(dset, m, pred_step, tpca=True, T=T, path='../Data/')

Xfold, y = Xfold.to(device), y.to(device)

n_nodes = Xfold.shape[1] // T
nEpochs = args.nEpochs

train_perc, valid_perc, test_perc = 0.2, 0.1, 0.7

idxTotal = torch.LongTensor(np.arange(nTotal)).to(device) # For time-series, keep temporal order

idxTrain = idxTotal[:np.floor(train_perc*nTotal).astype(int)]
idxValid = idxTotal[np.floor((train_perc)*nTotal).astype(int):np.floor((train_perc+valid_perc)*nTotal).astype(int)]
idxTest = idxTotal[np.floor((train_perc+valid_perc)*nTotal).astype(int):]

xTrain = Xfold[idxTrain]
xValid = Xfold[idxValid]
xTest = Xfold[idxTest] 
nTrain = idxTrain.shape[0]

yTrain = y[idxTrain]
yValid = y[idxValid]
yTest = y[idxTest]

Cval = None

C = torch.cov(xTrain.squeeze().T) # Compute covariance on current data

regressor = Regressor(in_size=T*n_nodes, h_size=args.h_size, h_size_2=args.h_size//2, out_size=n_nodes)
optimizer = optim.Adam(regressor.parameters(), lr=0.001)

batchSize = args.batchSize
nBatches = int(np.ceil(nTrain / batchSize))

eig_values, eig_vectors = torch.linalg.eig(C)
eig_values, eig_vectors = eig_values.real, eig_vectors.real
for i in range(eig_vectors.shape[0]):
    if eig_vectors[0,i]<0:
        eig_vectors[:,i] = -eig_vectors[:,i]

sortedEig, indices=torch.sort(eig_values, dim=0, descending=True, out=None)
eig_vectors = eig_vectors[:,indices]
if pca:
    X_reduced = torch.matmul(eig_vectors.T , xTrain.squeeze().T).T
    X_reduced_valid = torch.matmul(eig_vectors.T , xValid.squeeze().T).T
else:
    X_reduced = xTrain
    X_reduced_valid = xValid


MAE = nn.L1Loss()
MSE = nn.MSELoss()
Loss = nn.MSELoss()
epoch = 0
Best_Valid_Loss = 1e10
while epoch < nEpochs:
    C_old = None
    batch = 0 
    all_loss_train = 0

    while batch < nBatches:
        thisBatchIndices = torch.LongTensor(np.arange(nTrain)[batch * batchSize : (batch + 1) * batchSize]).to(device)
        xTrainBatch = X_reduced[torch.LongTensor(thisBatchIndices).to(device)] # B x G x N

        yTrainBatch = yTrain[thisBatchIndices]
        regressor.zero_grad()
        yHatTrainBatch = regressor(xTrainBatch)
        lossValueTrain = Loss(yHatTrainBatch.squeeze(), yTrainBatch.squeeze())
        lossValueTrain.backward()
        all_loss_train += lossValueTrain.detach()
        optimizer.step()
        batch+=1
    
    with torch.no_grad():

        yHatValid = regressor(X_reduced_valid)

        yHatValidInv = torch.tensor(Yscaler.inverse_transform(yHatValid.cpu()))
        yValidInv = torch.tensor(Yscaler.inverse_transform(yValid.cpu()))

        Valid_Loss = MSE(yHatValidInv.squeeze(), yValidInv.squeeze())
        
        if Valid_Loss < Best_Valid_Loss:
            Best_Valid_Loss = Valid_Loss
            best_model = deepcopy(regressor)

    print(f"""Epoch {epoch} train loss (MSE) {all_loss_train} val loss (MSE) {Valid_Loss}""")
    epoch+=1


if not online_test:
    #### 
    ## Compute true covariance 
    ####
    C = torch.cov(Xfold.squeeze().T) # Compute covariance on current data

    eig_values, eig_vectors = torch.linalg.eig(C)
    eig_values, eig_vectors = eig_values.real, eig_vectors.real
    for i in range(eig_vectors.shape[0]):
        if eig_vectors[0,i]<0:
            eig_vectors[:,i] = -eig_vectors[:,i]

    sortedEig, indices=torch.sort(eig_values, dim=0, descending=True, out=None)
    eig_vectors = eig_vectors[:,indices]
    if pca:
        X_reduced_test = torch.matmul(eig_vectors.T , xTest.squeeze().T).T
    else:
        X_reduced_test = xTest
    all_pred = regressor(X_reduced_test).detach()


else:

    all_pred = []
    all_loss = []

    mean = torch.cat([xTrain, xValid], dim=0).mean(0)
    prev_n = xTrain.shape[0] + xValid.shape[0]
    C_old = torch.cov(torch.cat([xTrain, xValid], dim=0).squeeze().T)

    testBatchSize = 1
    slidingWindowSize = 1

    nTest = yTest.shape[0]
    nBatches = int(np.ceil((nTest - testBatchSize) / slidingWindowSize)) + 1
    batchStart = 0
    print("Testing")
    regressor = deepcopy(best_model)
    optimizer = optim.Adam(regressor.parameters(), lr=0.001)
    C_dists = []    
    all_pred = []
    for n in tqdm(range(prev_n, nBatches+prev_n)):
        thisBatchIndices = torch.LongTensor(np.arange(nTest)[batchStart : batchStart + testBatchSize]).to(device)
        xTestBatch = xTest[torch.LongTensor(thisBatchIndices).to(device)] # B x G x N
        yTestBatch = yTest[thisBatchIndices]

        C, C_old, mean = update_covariance_mat(C_old, xTestBatch, gamma=gamma, mean=mean, stationary=False, norm=False)

        eig_values, eig_vectors = torch.linalg.eig(C)
        eig_values, eig_vectors = eig_values.real, eig_vectors.real
        for i in range(eig_vectors.shape[0]):
            if eig_vectors[0,i]<0:
                eig_vectors[:,i] = -eig_vectors[:,i]

        sortedEig, indices=torch.sort(eig_values, dim=0, descending=True, out=None)
        eig_vectors = eig_vectors[:,indices]

        xTestBatch = torch.matmul(eig_vectors.T , xTestBatch.squeeze())
        
        # Perform prediction and compute loss
        regressor.zero_grad()
        yHatTestBatch = regressor(xTestBatch)
        lossValueTest = Loss(yHatTestBatch.squeeze(), yTestBatch.squeeze())    
        all_pred.append(yHatTestBatch.detach())    

        lossValueTest.backward()
        optimizer.step()

        batchStart += slidingWindowSize


    all_pred = torch.stack(all_pred) 

    
yBestTest = torch.tensor(Yscaler.inverse_transform(all_pred.cpu()))
yTestInv = torch.tensor(Yscaler.inverse_transform(yTest.cpu()))
mae_test = MAE(yBestTest , yTestInv)
mape_test = sMAPE(yTestInv, yBestTest)
mse_test = MSE(yTestInv, yBestTest)

out_str = f"\n{dset},{pred_step},{args.lr},{args.optimizer},{T}," + \
        f"{gamma},{h_size},{mse_test},{mae_test},{mape_test}"

if write:
    with open(args.out_file, "a") as f:
        f.write(out_str)


print("Test results")
print(f"MAE {mae_test} MAPE {mape_test} MSE {mse_test}")
