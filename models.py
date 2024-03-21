
# SelectionGNN has been modified to accommodate regression problem
"""
architectures.py Architectures module

Definition of GNN architectures.

SelectionGNN: implements the GNN architecture
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

import graphML as gml
from layers import GraphFilter
# import Utils.graphTools

from utils import changeDataType

zeroTolerance = 1e-9 # Absolute values below this number are considered zero.

class ConvGNN(nn.Module):
    """
    SelectionGNN: implement the selection GNN architecture

    Initialization:

        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end
                     GSO, order = None, # Structure
                     coarsening = False)

        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            >> Obs.: If coarsening = True, this variable is ignored since the
                number of nodes in each layer is given by the graph coarsening
                algorithm.
            poolingFunction (nn.Module in Utils.graphML or in torch.nn): 
                summarizing function
            >> Obs.: If coarsening = True, then the pooling function is one of
                the regular 1-d pooling functions available in torch.nn (instead
                of one of the summarizing functions in Utils.graphML).
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            >> Obs.: If coarsening = True, then the pooling size is ignored 
                since, due to the binary tree nature of the graph coarsening
                algorithm, it always has to be 2.
                
            /** Readout layers **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
            coarsening (bool, default = False): if True uses graph coarsening
                instead of zero-padding to reduce the number of nodes.
            >> Obs.: (i) Graph coarsening only works when the number
                 of edge features is 1 -scalar weights-. (ii) The graph
                 coarsening forces a given order of the nodes, and this order
                 has to be used to reordering the GSO as well as the samples
                 during training; as such, this order is internally saved and
                 applied to the incoming samples in the forward call -it is
                 thus advised to use the identity ordering in the model class
                 when using the coarsening method-.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SelectionGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
        
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Time dimension
                 T = 0
                 ):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        self.T = T

        self.S = GSO
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.N = [GSO.shape[1]]
     
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.T, self.bias))
            # There is a 2*l below here, because we have two elements per
            # layer: graph filter and nonlinearity, so after each layer
            # we're actually adding elements to the (sequential) list.
            #\\ Nonlinearity -> not at last layer!
            gfl.append(self.sigma(0.1))
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers

        fc = []
        if len(self.dimLayersMLP) > 1:
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer coming
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
                self.init_weights(fc[-1])
                if l < len(dimLayersMLP) - 1: # No activation at last layer
                    fc.append(nn.LeakyReLU(0.1))

        self.MLP = nn.Sequential(*fc)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def changeGSO(self, GSO):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct shapes
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
            
        # Change data type and device as required
        self.S = deepcopy(GSO)
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Update GSO and LSIGF
        for l in range(self.L):
            self.GFL[2*l].addGSO(self.S) # Graph convolutional layer

    def get_weights(self):
        weights = []
        for l in range(self.L):
            weights.append(self.GFL[2*l].get_weights()) # Graph convolutional layer
        return weights
                    

    def forward(self, x, ret_emb=False):
        
        # Now we compute the forward call
        assert len(x.shape) == 3
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y_filtered = self.GFL(x)
        y = self.MLP(y_filtered.permute(0,2,1).reshape(-1,y_filtered.shape[1]))
        y = y.reshape((y_filtered.shape[0], y_filtered.shape[2]))

        if ret_emb:
            return y, y_filtered
        return y
                    


class GatedGraphRecurrentNN(nn.Module):
    """
   GatedGraphRecurrentNN: implements the (time, node, edge)-gated GRNN 
   architecture. It is a single-layer gated GRNN and the hidden state is 
   initialized at random drawing from a standard gaussian.
    
    Initialization:
        
        GatedGraphRecurrentNN(dimInputSignals, dimOutputSignals,
                            dimHiddenSignals, nFilterTaps, bias, # Filtering
                            nonlinearityHidden, nonlinearityOutput,
                            nonlinearityReadout, # Nonlinearities
                            dimReadout, # Local readout layer
                            dimEdgeFeatures, GSO, # Structure
                            gateType) # Gating
        
        Input:
            /** Graph convolutions **/
            dimInputSignals (int): dimension of the input signals
            dimOutputSignals (int): dimension of the output signals
            dimHiddenSignals (int): dimension of the hidden state
            nFilterTaps (list of int): a list with two elements, the first one
                is the number of filter taps for the filters in the hidden
                state equation, the second one is the number of filter taps
                for the filters in the output
            bias (bool): include bias after graph filter on every layer
            
            /** Activation functions **/
            nonlinearityHidden (torch.function): the nonlinearity to apply
                when computing the hidden state; it has to be a torch function,
                not a nn.Module
            nonlinearityOutput (torch.function): the nonlinearity to apply when
                computing the output signal; it has to be a torch function, not
                a nn.Module.
            nonlinearityReadout (nn.Module): the nonlinearity to apply at the
                end of the readout layer (if the readout layer has more than
                one layer); this one has to be a nn.Module, instead of just a
                torch function.
                
            /** Readout layer **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).
                
            /** Graph structure **/
            dimEdgeFeatures (int): number of edge features
            GSO (np.array): graph shift operator of choice.
            
            /** Gating **/
            gateType (string): 'time', 'node' or 'edge' gating
            
            
        Output:
            nn.Module with a gated GRNN architecture with the above specified
            characteristics
            
    Forward call:
        
        GatedGraphRecurrentNN(x)
        
        Input:
            x (torch.tensor): input data of shape
                batchSize x timeSamples x dimInputSignals x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the GRNN; 
                batchSize x timeSamples x dimReadout[-1] x numberNodes
        
    Other methods:
            
        y, yGNN = .splitForward(x): gives the output of the entire GRNN y,
        which has shape batchSize x timeSamples x dimReadout[-1] x numberNodes,
        as well as the output of the GRNN (i.e. before the readout layers), 
        yGNN of shape batchSize x timeSamples x dimInputSignals x numberNodes. 
        This can be used to isolate the effect of the graph convolutions from 
        the effect of the readout layer.
        
        y = .singleNodeForward(x, nodes): outputs the value of the last
        layer at a single node. x is the usual input of shape batchSize 
        x timeSamples x dimInputSignals x numberNodes. nodes is either a single
        node (int) or a collection of nodes (list or numpy.array) of length
        batchSize, where for each element in the batch, we get the output at
        the single specified node. The output y is of shape batchSize 
        x timeSamples x dimReadout[-1].
        
        .changeGSO(S): takes as input a new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the GatedGraphRecurrentNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one.
        >> Obs.: The number of nodes in the GSOs need not be the same.
            
        
    """
    def __init__(self,
                 # Graph filtering
                 dimInputSignals,
                 dimOutputSignals,
                 dimHiddenSignals,
                 nFilterTaps, bias,
                 # Nonlinearities
                 nonlinearityHidden,
                 nonlinearityOutput,
                 nonlinearityReadout, # nn.Module
                 # Local MLP in the end
                 dimReadout,
                 # Structure
                 GSO,
                 # Gating
                 gateType):
        # Initialize parent:
        super().__init__()
        
        # A list of two int, one for the number of filter taps (the computation
        # of the hidden state has the same number of filter taps)
        assert len(nFilterTaps) == 2
        
        # Store the values (using the notation in the paper):
        self.F = dimInputSignals # Number of input features
        self.G = dimOutputSignals # Number of output features
        self.H = dimHiddenSignals # NUmber of hidden features
        self.K = nFilterTaps # Filter taps
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearityHidden
        self.rho = nonlinearityOutput
        self.nonlinearityReadout = nonlinearityReadout
        self.dimReadout = dimReadout
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        self.E = GSO.shape[0] # Number of edge features
        self.N = GSO.shape[1]
        self.S = GSO
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        
        #\\\ Hidden State RNN \\\
        # Create the layer that generates the hidden state, and generate z0
        # The type of hidden layer depends on the type of gating
        assert gateType == 'time' or gateType == 'node' or gateType == 'edge'
        if gateType == 'time':
            self.hiddenState = gml.TimeGatedHiddenState(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        elif gateType == 'node':
            self.hiddenState = gml.NodeGatedHiddenState(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        elif gateType == 'edge':
            self.hiddenState = gml.EdgeGatedHiddenState(self.F, self.H, self.K[0],
                                       nonlinearity = self.sigma, E = self.E,
                                       bias = self.bias)
        #\\\ Output Graph Filters \\\
        self.outputState = gml.GraphFilter(self.H, self.G, self.K[1],
                                              E = self.E, bias = self.bias)
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(self.S)
        self.outputState.addGSO(self.S)
        
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0: # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph 
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.G, dimReadout[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimReadout)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.nonlinearityReadout())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def splitForward(self, x):

        # Check the dimensions of the input
        #   S: E x N x N
        #   x: B x T x F[0] x N
        assert len(self.S.shape) == 3
        assert self.S.shape[0] == self.E     
        N = self.S.shape[1]
        assert self.S.shape[2] == N
        
        assert len(x.shape) == 4
        B = x.shape[0]
        T = x.shape[1]
        assert x.shape[2] == self.F
        assert x.shape[3] == N
        
        # This can be generated here or generated outside of here, not clear yet
        # what's the most coherent option
        z0 = torch.randn((B, self.H, N), device = x.device)
        
        # Compute the trajectory of hidden states
        z, _ = self.hiddenState(x, z0)
        z = z.reshape((B*T,self.H,N))
        # Compute the output trajectory from the hidden states
        yOut = self.outputState(z)
        yOut = self.rho(yOut) # Don't forget the nonlinearity!
        yOut = yOut.reshape((B,T,self.G,N))
        #   B x T x G x N
        # Change the order, for the readout
        y = yOut.permute(0, 1, 3, 2) # B x T x N x G
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x T x N x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 1, 3, 2), yOut
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output
        
    def singleNodeForward(self, x, nodes):
        
        # x is of shape B x T x F[0] x N
        batchSize = x.shape[0]
        N = x.shape[3]
        
        # nodes is either an int, or a list/np.array of ints of size B
        assert type(nodes) is int \
                or type(nodes) is list \
                or type(nodes) is np.ndarray
        
        # Let us start by building the selection matrix
        # This selection matrix has to be a matrix of shape
        #   B x 1 x N[-1] x 1
        # so that when multiplying with the output of the forward, we get a
        #   B x T x dimRedout[-1] x 1
        # and we just squeeze the last dimension
        
        # TODO: The big question here is if multiplying by a matrix is faster
        # than doing torch.index_select
        
        # Let's always work with numpy arrays to make it easier.
        if type(nodes) is int:
            # Change the node number to accommodate the new order
            nodes = self.order.index(nodes)
            # If it's int, make it a list and an array
            nodes = np.array([nodes], dtype=np.int)
            # And repeat for the number of batches
            nodes = np.tile(nodes, batchSize)
        if type(nodes) is list:
            newNodes = [self.order.index(n) for n in nodes]
            nodes = np.array(newNodes, dtype = np.int)
        elif type(nodes) is np.ndarray:
            newNodes = np.array([np.where(np.array(self.order) == n)[0][0] \
                                                                for n in nodes])
            nodes = newNodes.astype(np.int)
        # Now, nodes is an np.int np.ndarray with shape batchSize
        
        # Build the selection matrix
        selectionMatrix = np.zeros([batchSize, 1, N, 1])
        selectionMatrix[np.arange(batchSize), nodes, 0] = 1.
        # And convert it to a tensor
        selectionMatrix = torch.tensor(selectionMatrix,
                                       dtype = x.dtype,
                                       device = x.device)
        
        # Now compute the output
        y = self.forward(x)
        # This output is of size B x T x dimReadout[-1] x N
        
        # Multiply the output
        y = torch.matmul(y, selectionMatrix)
        #   B x T x dimReadout[-1] x 1
        
        # Squeeze the last dimension and return
        return y.squeeze(3)
    
    def changeGSO(self, GSO):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
            
        self.S = GSO
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Add the GSO for each graph filter
        self.hiddenState.addGSO(self.S)
        self.outputState.addGSO(self.S)
        

class Regressor(nn.Module):

    def __init__(self, in_size=32, h_size=32, out_size=1):
        super(Regressor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_size, h_size),
            nn.LeakyReLU(0.1),
            nn.Linear(h_size, out_size)
        )
        self.init_weights(self.mlp)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.mlp(x)


class myLSTM(nn.Module):

    def __init__(self, in_size=32, h_size=32, num_layers=1, out_size=1):
        super(myLSTM, self).__init__()

        self.lstm = nn.LSTM(in_size, h_size, num_layers, batch_first=True)

        self.mlp = nn.Linear(h_size, out_size)

        self.init_weights(self.lstm)
        self.init_weights(self.mlp)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.mlp(x[:,-1])
    


class VNNLSTM(nn.Module):

    def __init__(self, dimNodeSignals,nFilterTaps, 
             bias, nonlinearity,nSelectedNodes,poolingFunction, 
             poolingSize, dimLayersMLP, GSO, in_size=32, h_size=32, 
             num_layers=1, out_size=1):
        super(VNNLSTM, self).__init__()

        self.gnn = VNN(dimNodeSignals=dimNodeSignals, 
             nFilterTaps=nFilterTaps, 
             bias=bias, 
             nonlinearity=nonlinearity, 
             nSelectedNodes=nSelectedNodes, 
             poolingFunction=poolingFunction, 
             poolingSize=poolingSize, 
             dimLayersMLP=dimLayersMLP, 
             GSO=GSO)

        self.lstm = nn.LSTM(in_size, h_size, num_layers, batch_first=True)

        self.mlp = nn.Linear(h_size, out_size)

        self.init_weights(self.lstm)
        self.init_weights(self.mlp)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def changeGSO(self, S):
        self.gnn.changeGSO(S)

    def forward(self, x):
        x = self.gnn(x.permute(1,0,2))
        x, _ = self.lstm(x.permute(1,0,2))
        return self.mlp(x[:,-1])



class VNN(nn.Module):
    """
    SelectionGNN: implement the selection GNN architecture

    Initialization:

        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end
                     GSO, order = None, # Structure
                     coarsening = False)

        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            >> Obs.: If coarsening = True, this variable is ignored since the
                number of nodes in each layer is given by the graph coarsening
                algorithm.
            poolingFunction (nn.Module in Utils.graphML or in torch.nn): 
                summarizing function
            >> Obs.: If coarsening = True, then the pooling function is one of
                the regular 1-d pooling functions available in torch.nn (instead
                of one of the summarizing functions in Utils.graphML).
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            >> Obs.: If coarsening = True, then the pooling size is ignored 
                since, due to the binary tree nature of the graph coarsening
                algorithm, it always has to be 2.
                
            /** Readout layers **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
            coarsening (bool, default = False): if True uses graph coarsening
                instead of zero-padding to reduce the number of nodes.
            >> Obs.: (i) Graph coarsening only works when the number
                 of edge features is 1 -scalar weights-. (ii) The graph
                 coarsening forces a given order of the nodes, and this order
                 has to be used to reordering the GSO as well as the samples
                 during training; as such, this order is internally saved and
                 applied to the incoming samples in the forward call -it is
                 thus advised to use the identity ordering in the model class
                 when using the coarsening method-.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SelectionGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
        
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Ordering
                 order = None,
                 # Coarsening
                 coarsening = False):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features     
        self.S = GSO
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        self.alpha = poolingSize
        self.coarsening = False # If it failed because there are more than
        # one edge feature, then just set this to false, so we do not
        # need to keep checking whether self.E == 1 or not, just this
        # one
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ Final aggregation layer \\\
        fc = []
        if len(self.dimLayersMLP) > 1:
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer coming
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
                # self.init_weights(fc[-1])
                if l < len(dimLayersMLP) - 1: # No activation at last layer
                    fc.append(nn.LeakyReLU(0.1))

        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
            
        # Now, if we don't have coarsening, then we need to reorder the GSO,
        # and since this GSO reordering will affect several parts of the non
        # coarsening algorithm, then we will do it now
        # Reorder the GSO
        self.S = GSO
        # Change data type and device as required
        self.S = changeDataType(self.S, dataType)
        if device is not None:
            self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0:
            # (If it's coarsening, then the pooling size cannot change)
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes (this only makes sense
        # if there is no coarsening, because if it is coarsening, the list with
        # the number of nodes to be considered is ignored.)
        if len(nSelectedNodes) > 0:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.GFL[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                           self.alpha[l])
                self.GFL[3*l+2].addGSO(self.S)
        elif len(nSelectedNodes) == 0:
            # Just update the GSO
            for l in range(self.L):
                self.GFL[3*l+2].addGSO(self.S)
        
        # And update in the LSIGF that is still missing (recall that the
        # ordering for the non-coarsening case has already been done)
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S) # Graph convolutional layer

    def splitForward(self, x):
        
        # Reorder the nodes from the data
        # If we have added dummy nodes (which, has to happen when the size
        # is different and we chose coarsening), then we need to use the
        # provided permCoarsening function (which acts on data to add dummy
        # variables)
        # Now we compute the forward call
        assert len(x.shape) == 3
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        # yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        # print(yFlat.shape)
        return self.MLP(y.permute(0,2,1)), y
        # If self.MLP is a sequential on an empty list it just does nothing.
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S)
            self.GFL[3*l+2].addGSO(self.S)
