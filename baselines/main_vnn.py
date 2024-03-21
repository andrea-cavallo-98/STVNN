import sys 
sys.path.append('../')
import pandas as pd
import numpy as np
import torch 
from torch import nn, optim
from models import ConvGNN, VNN
from copy import deepcopy
from utils import *
from tqdm import tqdm
import graphML as gml

args = parse_args()

skip_val = False
# dset = f"NOA"
dset = args.dset
stationary = dset.startswith("synth")

m = args.m
pred_step = args.pred_step # how many steps in the future we want to predict
if torch.cuda.is_available():
    print("Using CUDA")
    device = "cuda"
else:
    device = "cpu"


T = args.T
gamma = args.gamma
dimNodeSignals = args.dimNodeSignals
L = len(dimNodeSignals) - 1
nFilterTaps = [args.filter_taps] * L
dimLayersMLP = args.dimLayersMLP 
online = args.online
write = args.out_file is not None
lr = args.lr

Xfold, nTotal, y, Yscaler = load_and_normalize_data(dset, m, pred_step, rec=True, T=T, path='../Data/')

Xfold, y = Xfold.to(device).squeeze(), y.to(device)
n_nodes = Xfold.shape[2]


nEpochs = args.nEpochs

# Split indices
if online:
    train_perc, valid_perc, test_perc = 0.2, 0.1, 0.7
else:
    train_perc, valid_perc, test_perc = 0.7, 0.1, 0.2

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
C = torch.cov(xTrain[:,-1].squeeze().T) # Compute covariance on current data
C = C / torch.trace(C) # trace-normalize to avoid numerical issues

# GNN = ConvGNN(dimNodeSignals=dimNodeSignals, # hidden size
#                                 nFilterTaps=nFilterTaps, # k, i.e. neighborhoods (len = len(dimNodeSignals)-1)
#                                 bias=True, 
#                                 nonlinearity=nn.LeakyReLU, 
#                                 dimLayersMLP=dimLayersMLP, 
#                                 GSO=C,
#                                 T=T).to(device)

GNN = VNN(dimNodeSignals=dimNodeSignals, 
             nFilterTaps=nFilterTaps, 
             bias=True, 
             nonlinearity=nn.ReLU, 
             nSelectedNodes=[n_nodes,n_nodes], 
             poolingFunction=gml.NoPool, 
             poolingSize=[1,1], 
             dimLayersMLP=dimLayersMLP, 
             GSO=C)


batchSize = args.batchSize
nTrainBatches = int(np.ceil(nTrain / batchSize))

Loss = nn.MSELoss()
MAE = nn.L1Loss()
MSE = nn.MSELoss()
if args.optimizer == "Adam":
    optimizer = optim.Adam(GNN.parameters(), lr=lr, weight_decay=0.001)
elif args.optimizer == "SGD":
    optimizer = optim.SGD(GNN.parameters(), lr=lr)
else:
    print("Optimizer not allowed")
    raise Exception()

Best_Valid_Loss, Best_Valid_MAPE = 1e10, 1e10
for epoch in range(nEpochs):
    C_old = None
    mean = torch.zeros(xTrain.shape[2]).to(device)
    tot_train_loss = []
    tot_val_mae = 0.
    tot_val_mape = 0.
    train_perm_idx = torch.randperm(nTrainBatches).to(device) # shuffle order during training

    for batch in tqdm(range(nTrainBatches)):
        thisBatchIndices = torch.LongTensor(np.arange(nTrain)[batch * batchSize : (batch + 1) * batchSize]).to(device)
        xTrainBatch = xTrain[thisBatchIndices]
        yTrainBatch = yTrain[thisBatchIndices]
        C, C_old, mean = update_covariance_mat(C_old, xTrainBatch[:,-1], gamma=gamma, 
                                               mean=mean, stationary=stationary, n=batch*batchSize)
        GNN.changeGSO(C)

        GNN.zero_grad()
        yHatTrainBatch = GNN(xTrainBatch)
        lossValueTrain = Loss(yHatTrainBatch.squeeze(), yTrainBatch.squeeze())
        lossValueTrain.backward()
        optimizer.step()
        tot_train_loss.append(lossValueTrain.detach())
        
    all_pred = []
    if skip_val:
        Best_GNN = deepcopy(GNN)
    else:
        with torch.no_grad():
            prev_n = xTrain.shape[0]
            nValid = xValid.shape[0]
            nValidBatches = int(np.ceil(nValid / batchSize))
            for batch in range(nValidBatches):
                thisBatchIndices = torch.LongTensor(np.arange(nValid)[batch * batchSize : (batch + 1) * batchSize]).to(device)
                xValidBatch = xValid[thisBatchIndices]

                C, C_old, mean = update_covariance_mat(C_old, xValidBatch[:,-1], gamma=gamma, 
                                                       mean=mean, stationary=stationary, n=prev_n+batch*batchSize)
                GNN.changeGSO(C.squeeze())

                yHatValid = GNN(xValidBatch)
                all_pred.append(yHatValid.detach().squeeze(2))

            all_pred = torch.cat(all_pred, dim=0) 
            yHatValid = torch.tensor(Yscaler.inverse_transform(all_pred.cpu()))
            yValidInv = torch.tensor(Yscaler.inverse_transform(yValid.cpu()))

            Valid_Loss = MAE(yHatValid.squeeze(), yValidInv.squeeze())
            valid_MAPE = sMAPE(yValidInv.squeeze(), yHatValid.squeeze())

            tot_val_mae += Valid_Loss.detach()
            tot_val_mape += valid_MAPE.detach()

            if tot_val_mae < Best_Valid_Loss:
                Best_Valid_Loss = tot_val_mae
                Best_GNN = deepcopy(GNN)

            print(f"""Epoch {epoch} train loss (MSE) {sum(tot_train_loss)} val loss (MAE) {tot_val_mae} val MAPE {tot_val_mape}""")
    
all_pred = []
all_pred_offline = []

GNN = deepcopy(Best_GNN)

mean = torch.cat([xTrain[:,-1], xValid[:,-1]], dim=0).mean(0)
prev_n = xTrain.shape[0] + xValid.shape[0]
C_old = torch.cov(torch.cat([xTrain[:,-1], xValid[:,-1]], dim=0).squeeze().T)

if args.optimizer == "Adam":
    optimizer = optim.Adam(GNN.parameters(), lr=lr, weight_decay=0.001)
elif args.optimizer == "SGD":
    optimizer = optim.SGD(GNN.parameters(), lr=lr)

if online:
    testBatchSize = 1
else:
    testBatchSize = args.batchSize

nTest = yTest.shape[0]
nTestBatches = int(np.ceil(nTest / testBatchSize))
all_test_loss = []
print("Testing")
for batch in tqdm(range(nTestBatches)):
    thisBatchIndices = torch.LongTensor(np.arange(nTest)[batch * batchSize : (batch + 1) * batchSize]).to(device)
    xTestBatch = xTest[thisBatchIndices]
    yTestBatch = yTest[thisBatchIndices]
    C, C_old, mean = update_covariance_mat(C_old, xTestBatch[:,-1], gamma=gamma, 
                                           mean=mean, stationary=stationary, n=prev_n+batch*batchSize)
    GNN.changeGSO(C)

    GNN.zero_grad()
    yHatTestBatch = GNN(xTestBatch)
    lossValueTest = Loss(yHatTestBatch.squeeze(), yTestBatch.squeeze())
    lossValueTest.backward()
    optimizer.step()
    all_pred.append(yHatTestBatch.detach().squeeze(2))
    all_test_loss.append(lossValueTest.detach())

if dset.startswith("synth"):
    dset = "synth"

all_pred = torch.cat(all_pred, dim=0) 
yBestTest = torch.tensor(Yscaler.inverse_transform(all_pred.cpu()))
yTestInv = torch.tensor(Yscaler.inverse_transform(yTest.cpu()))

# torch.save(GNN.state_dict(), f"trained_models/TVNN_{dset}.pt")
# torch.save(args, f"trained_models/args_{dset}")
# torch.save(yBestTest, f"trained_models/pred_{dset}.pt")

mae_test = MAE(yBestTest , yTestInv)
mse_test = MSE(yBestTest , yTestInv)
mape_test = sMAPE(yTestInv, yBestTest)

out_str = f"\n{dset},{pred_step},{lr},{args.optimizer},{T},{' '.join(map(str, dimNodeSignals))}," + \
        f"{gamma},{args.filter_taps},{mse_test},{mae_test},{mape_test}"

if write:
    with open(args.out_file, "a") as f:
        f.write(out_str)

print("MSE test: ", mse_test, "MAE test: ", mae_test, "sMAPE test: ", mape_test)

