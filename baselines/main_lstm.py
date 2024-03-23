import sys 
sys.path.append('../')

import pandas as pd
import numpy as np
import torch 
from torch import nn, optim
from copy import deepcopy
from utils import *
from tqdm import tqdm
from models import myLSTM

args = parse_args()

dset = args.dset

m = args.m
pred_step = args.pred_step # how many steps in the future we want to predict

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

T = args.T
online = args.online
write = args.out_file is not None

Xfold, nTotal, y, Yscaler = load_and_normalize_data(dset, m, pred_step, rec=True, T=T, path='../Data/')
Xfold, y = Xfold.to(device).squeeze(), y.to(device)

n_nodes = Xfold.shape[2]

Loss = nn.MSELoss()
nEpochs = args.nEpochs

# Split indices
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


model = myLSTM(in_size=n_nodes, h_size=2*n_nodes, num_layers=2, out_size=n_nodes)

batchSize = args.batchSize
nBatches = int(np.ceil(nTrain / batchSize))

MAE = nn.L1Loss()
MSE = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch = 0
Best_Valid_Loss = 100
for epoch in range(nEpochs):
    C_old = None
    all_train_loss = 0

    for batch in tqdm(range(nBatches)):
        thisBatchIndices = torch.LongTensor(np.arange(nTrain)[batch * batchSize : (batch + 1) * batchSize]).to(device)
        xTrainBatch = xTrain[thisBatchIndices]
        yTrainBatch = yTrain[thisBatchIndices]

        model.zero_grad()
        yHatTrainBatch = model(xTrainBatch)
        lossValueTrain = Loss(yHatTrainBatch.squeeze(), yTrainBatch.squeeze())
        lossValueTrain.backward()
        all_train_loss += lossValueTrain.detach()
        optimizer.step()

    with torch.no_grad():

        yHatValid = model(xValid)
        Valid_Loss = MAE(yHatValid.squeeze(), yValid.squeeze())
        valid_MAPE = MAPE(yValid.squeeze(), yHatValid.squeeze())

        if Valid_Loss < Best_Valid_Loss:
            Best_Valid_Loss = Valid_Loss
            best_model = deepcopy(model)

    print(f"""Epoch {epoch} train loss (MSE) {all_train_loss} val loss (MAE) {Valid_Loss} val MAPE {valid_MAPE}""")
    

all_pred = []
model = deepcopy(best_model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
testBatchSize = 1
slidingWindowSize = 1
nTest = yTest.shape[0]
nBatches = int(np.ceil((nTest - testBatchSize) / slidingWindowSize)) + 1
batchStart = 0
for _ in tqdm(range(nBatches)):

    thisBatchIndices = torch.LongTensor(np.arange(nTest)[batchStart : batchStart + testBatchSize]).to(device)
    xTestBatch = xTest[thisBatchIndices] # B x G x N
    yTestBatch = yTest[thisBatchIndices]

    model.zero_grad()
    yHatTestBatch = model(xTestBatch)
    if batchStart == 0:
        all_pred.append(yHatTestBatch.squeeze().detach().reshape((-1, n_nodes)))
    else: 
        all_pred.append(yHatTestBatch[testBatchSize-slidingWindowSize:].squeeze().detach().reshape((-1, n_nodes)))
    lossValueTest = Loss(yHatTestBatch.squeeze(), yTestBatch.squeeze())
    lossValueTest.backward()
    optimizer.step()
    batchStart += slidingWindowSize

all_pred = torch.cat(all_pred, dim=0)
yBestTest = torch.tensor(Yscaler.inverse_transform(all_pred.cpu()))
yTestInv = torch.tensor(Yscaler.inverse_transform(yTest.cpu()))

mae_test = MAE(yBestTest , yTestInv)
mse_test = MSE(yBestTest , yTestInv)
mape_test = sMAPE(yTestInv, yBestTest)

out_str = f"\n{dset},{pred_step},{T}," + \
        f"{mse_test},{mae_test},{mape_test}"

if write:
    with open(args.out_file, "a") as f:
        f.write(out_str)


print(f"Test MSE: {mse_test.detach().item()}\nTest MAE: {mae_test.detach().item()}\nTest sMAPE: {mape_test.detach().item()}")


