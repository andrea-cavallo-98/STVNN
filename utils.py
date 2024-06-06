from sklearn.preprocessing import StandardScaler
import argparse
import torch
import numpy as np
import pandas as pd

def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """
    
    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.
    
    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.
    
    # If we can't recognize the type, we just make everything numpy.
    
    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype
    
    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)
            
    # This only converts between numpy and torch. Any other thing is ignored
    return x


def MAPE(yTrue, yPred):
    return torch.mean(torch.abs((yTrue - yPred) / yTrue))    

def sMAPE(yTrue, yPred): 
    return torch.mean(torch.abs(yPred - yTrue) / ((torch.abs(yPred) + torch.abs(yTrue))/2)) 



def normalize_data_synth(data, m, pred_step):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    # Normalize
    x_input = data[:m,:].numpy()
    y_output = data[pred_step:m+pred_step,:].numpy()
    Xnorm = torch.tensor(Xscaler.fit_transform(x_input).T).float() # normalized input

    Xfold = Xnorm.T.unsqueeze(1) # B x G x N
    nTotal = y_output.shape[0]
    y = torch.tensor(Yscaler.fit_transform(y_output)).float() 
    
    return Xfold, nTotal, y, Yscaler


def normalize_data_conv(df, m, pred_step):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    # Normalize
    x_input = df.iloc[:m,:].to_numpy()
    y_output = df.iloc[pred_step:m+pred_step,:].to_numpy()
    nTotal = y_output.shape[0]
    y = torch.tensor(Yscaler.fit_transform(y_output)).float() 
    Xnorm = torch.tensor(Xscaler.fit_transform(x_input).T).float() # normalized input

    Xfold = Xnorm.T.unsqueeze(1) # B x G x N
    
    return Xfold, nTotal, y, Yscaler


def normalize_data_rec(df, m, pred_step, T):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    # Normalize
    x_input = df.iloc[:m,:].to_numpy()
    y_output = df.iloc[T+pred_step-1:m+pred_step-1,:].to_numpy()
    Xnorm = torch.tensor(Xscaler.fit_transform(x_input).T).float() # normalized input
    Xfold = Xnorm.unfold(1, T, 1)[:,:-pred_step,:] # Input features, shape (Nnodes, m, T)
    Xfold = Xfold.permute(1, 2, 0).unsqueeze(2) # B x T x F[0] x N
    nTotal = y_output.shape[0] - T
    y = torch.tensor(Yscaler.fit_transform(y_output)).float() 
    
    return Xfold, nTotal, y, Yscaler

def normalize_data_rec_synth(data, m, pred_step, T):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    # Normalize
    x_input = data[:m,:].numpy()
    y_output = data[T+pred_step-1:m+pred_step-1,:].numpy()
    Xnorm = torch.tensor(Xscaler.fit_transform(x_input).T).float() # normalized input

    Xfold = Xnorm.unfold(1, T, 1)[:,:-pred_step,:] # Input features, shape (Nnodes, m, T)
    Xfold = Xfold.permute(1, 2, 0).unsqueeze(2) # B x T x F[0] x N
    nTotal = y_output.shape[0] - T
    y = torch.tensor(Yscaler.fit_transform(y_output)).float() 
    
    return Xfold, nTotal, y, Yscaler

def normalize_data_tpca_pt(data, m, pred_step, T):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    n_nodes = data.shape[1]

    # Normalize
    x_input = data[:m,:].numpy()
    y_output = data[T+pred_step-1:m+pred_step,:].numpy()
    Xnorm = torch.tensor(Xscaler.fit_transform(x_input).T).float() # normalized input

    Xfold = Xnorm.unfold(1, T, 1)[:,:-pred_step,:] # Input features, shape (Nnodes, m, T)
    Xt = Xfold.permute((2,0,1)).reshape((n_nodes*T,-1)).T
    nTotal = y_output.shape[0] - T
    y = torch.tensor(Yscaler.fit_transform(y_output)).float() 
    return Xt, nTotal, y, Yscaler

def normalize_data_tpca_pd(df, m, pred_step, T):
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    # Normalize
    x_input = df.iloc[:m,:].to_numpy()
    y_output = df.iloc[T+pred_step-1:m+pred_step-1,:].to_numpy()
    Xnorm = torch.tensor(Xscaler.fit_transform(x_input).T).float() # normalized input
    Xfold = Xnorm.unfold(1, T, 1)[:,:-pred_step,:] # Input features, shape (Nnodes, m, T)
    n_nodes = Xfold.shape[0]
    Xt = Xfold.permute((2,0,1)).reshape((n_nodes*T,-1)).T
    nTotal = y_output.shape[0] - T
    y = torch.tensor(Yscaler.fit_transform(y_output)).float() 
    
    return Xt, nTotal, y, Yscaler


def parse_boolean(value):
    """Parse boolean values passed as argument"""
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False

def parse_hidden_sizes(value):
    """Create list of int from string of csv"""
    
    return list(map(lambda x: int(x), value.split(",")))

def update_covariance_mat(C_old, xTrainBatch, gamma, mean=None, stationary=True, n=10, norm=True):
    xTrainBatch = xTrainBatch.squeeze(1)
    if not stationary:
        if mean is not None:
            C_new = torch.matmul((xTrainBatch - mean).T,xTrainBatch - mean) / xTrainBatch.shape[0]
            if C_old is not None:
                mean = (1-gamma)*mean + gamma * xTrainBatch.mean(0) 
                Craw = gamma * C_new + (1-gamma) * C_old
            else: # First pass, no C_old
                mean = xTrainBatch.mean(0)
                Craw = C_new
        else:
            if C_old is not None:
                Craw = gamma * torch.cov(xTrainBatch.squeeze().T) + (1-gamma) * C_old
            else: # First pass, no C_old
                Craw = torch.cov(xTrainBatch.squeeze().T)
    else:
        t = n
        if mean is not None:
            C_new = torch.matmul((xTrainBatch - mean).T,xTrainBatch - mean) / xTrainBatch.shape[0]
            if C_old is not None:
                Craw = (t - 1) / t * C_old + 1 / (t + 1) * C_new
                mean = t / (t + 1) * mean + 1 / (t + 1) * xTrainBatch.mean(0)
            else: # First pass, no C_old
                mean = xTrainBatch.mean(0)
                Craw = C_new
        else:
            if C_old is not None:
                Craw = gamma * torch.cov(xTrainBatch.squeeze().T) + (1-gamma) * C_old
            else: # First pass, no C_old
                Craw = torch.cov(xTrainBatch.squeeze().T)


    if torch.trace(Craw) != 0 and norm:
        C = Craw / torch.trace(Craw)
    else:
        C = Craw
    
    return C, Craw, mean


def selectCurrentBatch(batch, batchSize, L, T, N, device, X, Y):
    thisBatchIndices = torch.LongTensor(np.arange(N)[batch * batchSize : (batch + 1) * batchSize]).to(device)
    if batch * batchSize < L * (T - 1):
        thisBatchIndicesPad = np.concatenate(
            (
                np.ones([L * (T - 1),]) * batch * batchSize, # No past values are available, just replicate the first value
                np.arange(N)[batch * batchSize : (batch + 1) * batchSize] 
                )
        )
    else:
        thisBatchIndicesPad = np.arange(N)[batch * batchSize - L * (T - 1) : (batch + 1) * batchSize] # Needed for temporal aggregation
    XBatch = X[torch.LongTensor(thisBatchIndicesPad).to(device)] # B x G x N
    YBatch = Y[thisBatchIndices]
    return XBatch, YBatch

def parse_args():
    """ Parse arguments """
    parse = argparse.ArgumentParser()

    ## Run details
    parse.add_argument("--m", help="number of time samples", type=int, default=1000000)
    parse.add_argument("--pred_step", help="prediction step", type=int, default=1)
    parse.add_argument("--T", help="history length", type=int, default=5)
    parse.add_argument("--update_covariance", help="whether to perform covariance update", type=parse_boolean, default=False)
    parse.add_argument("--gamma", help="covariance update coefficient", type=float, default=0.1)
    parse.add_argument("--dimNodeSignals", help="sizes of GNN hidden layers", type=parse_hidden_sizes, default=[1,8,8])
    parse.add_argument("--dimLayersMLP", help="sizes of MLP hidden layers", type=parse_hidden_sizes, default=[8,1])
    # NOTE: filter taps is implemented such that if this parameter is set to
    # e.g. 3, then the filter has order 2
    parse.add_argument("--filter_taps", help="filter taps", type=int, default=2)
    parse.add_argument("--online", help="whether to update the model online", type=parse_boolean, default=True)
    parse.add_argument("--stationary", help="whether to use the stationary or non-stationary covariance update", type=parse_boolean, default=False)
    parse.add_argument("--nEpochs", help="epochs", type=int, default=1)
    parse.add_argument("--batchSize", help="batch size", type=int, default=1)
    parse.add_argument("--dimOutputSignals", help="dim recurrent output", type=int, default=8)
    parse.add_argument("--dimHiddenSignals", help="dim recurrent hidden state", type=int, default=8)
    parse.add_argument("--out_file", help="output file", type=str, default=None)
    parse.add_argument("--dset", help="dataset", type=str, default="NOA")
    parse.add_argument("--optimizer", help="optimizer", type=str, default="SGD")
    parse.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parse.add_argument("--lr_test", help="learning rate for online test", type=float, default=0.001)
    parse.add_argument("--h_size", help="MLP size for TPCA", type=int, default=256)
    parse.add_argument("--suffix", help="suffix for some experiments", type=str, default="")

    args = parse.parse_args()
    return args


def load_and_normalize_data(dset, m, pred_step, rec=False, T=1, path='Data/', tpca=False):

    if dset.startswith("synth"):
        data = torch.load(f"{path}{dset}")
    elif dset == "exchange_rate":
        df = pd.read_csv(path + dset + ".txt")
    elif dset.startswith("NOA"):
        data = torch.tensor(np.load(f"{path}NOA_109_data.npy").T)
    elif dset == "Molene":
        data = torch.tensor(np.load(f"{path}molene.npy"))

    if tpca:
        if dset == "exchange_rate":
            Xfold, nTotal, y, Yscaler = normalize_data_tpca_pd(df, m, pred_step, T)
        elif dset.startswith("synth") or dset.startswith("NOA") or dset == "Molene":
            Xfold, nTotal, y, Yscaler = normalize_data_tpca_pt(data, m, pred_step, T)
    elif not rec:
        if dset == "exchange_rate":
            Xfold, nTotal, y, Yscaler = normalize_data_conv(df, m, pred_step)
        elif dset.startswith("synth") or dset.startswith("NOA") or dset == "Molene":
            Xfold, nTotal, y, Yscaler = normalize_data_synth(data, m, pred_step)
    else:
        if dset == "exchange_rate":
            Xfold, nTotal, y, Yscaler = normalize_data_rec(df, m, pred_step, T)
        elif dset.startswith("synth") or dset.startswith("NOA") or dset == "Molene":
            Xfold, nTotal, y, Yscaler = normalize_data_rec_synth(data, m, pred_step, T)

    return Xfold, nTotal, y, Yscaler