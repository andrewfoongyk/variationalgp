# perform full, exact GP regression
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""this implementation follows Algorithm 2.1 in Rasmussen and Williams"""

class GP(nn.Module):
    def __init__(self, no_inputs):
        super(GP, self).__init__()
        # initialise hyperparameters
        self.no_inputs = no_inputs # input dimension
        self.logsigmaf2 = nn.Parameter(torch.Tensor([0])) # function variance
        self.logl2 = nn.Parameter(torch.zeros(no_inputs)) # horizontal length scales
        self.logsigman2 = nn.Parameter(torch.Tensor([0])) # noise variance

    def get_LL(self, train_inputs, train_outputs):
        # form the kernel matrix Knn using squared exponential ARD kernel
        train_inputs_col = torch.unsqueeze(train_inputs.transpose(0,1), 2)
        train_inputs_row = torch.unsqueeze(train_inputs.transpose(0,1), 1)
        squared_distances = (train_inputs_col - train_inputs_row)**2        
        length_factors = (1/(2*torch.exp(self.logl2))).reshape(self.no_inputs,1,1)        
        Knn = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))
        
        # cholesky decompose
        L = torch.potrf(Knn + torch.exp(self.logsigman2)*torch.eye(train_inputs.shape[0]), upper=False) # lower triangular decomposition
        Lslashy = torch.trtrs(train_outputs, L, upper=False)[0]
        alpha = torch.trtrs(Lslashy, torch.transpose(L,0,1))[0]

        # get log marginal likelihood
        LL = -0.5*torch.dot(train_outputs, torch.squeeze(alpha)) - torch.sum(torch.log(torch.diag(L))) - (train_inputs.shape[0]/2)*torch.log(torch.Tensor([2*3.1415926536]))
        return LL
    
    def posterior_predictive(self, train_inputs, train_outputs, test_inputs):
        # form the kernel matrix Knn using squared exponential ARD kernel
        train_inputs_col = torch.unsqueeze(train_inputs.transpose(0,1), 2)
        train_inputs_row = torch.unsqueeze(train_inputs.transpose(0,1), 1)
        squared_distances = (train_inputs_col - train_inputs_row)**2        
        length_factors = (1/(2*torch.exp(self.logl2))).reshape(self.no_inputs,1,1)        
        Knn = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))
        
        no_test = test_inputs.shape[0]
        pred_mean = torch.zeros(no_test)
        pred_var = torch.zeros(no_test)

        # cholesky decompose
        L = torch.potrf(Knn + torch.exp(self.logsigman2)*torch.eye(train_inputs.shape[0]), upper=False) # lower triangular decomposition
        Lslashy = torch.trtrs(train_outputs, L, upper=False)[0]
        alpha = torch.trtrs(Lslashy, torch.transpose(L,0,1))[0]

        # get mean and predictive variance for each test point
        for i in range(no_test):
            # form the test point kernel vector
            
            squared_distances = ((test_inputs[i]).reshape(self.no_inputs,1,1) - train_inputs_col)**2        
            length_factors = (1/(2*torch.exp(self.logl2))).reshape(self.no_inputs,1,1)
            kstar = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))
            
            # get predictive mean
            pred_mean[i] = torch.squeeze(kstar.transpose(0,1) @ alpha)

            # get predictive variance
            v = torch.squeeze(torch.trtrs(kstar, L, upper=False)[0])
            pred_var[i] = torch.exp(self.logsigmaf2) - torch.dot(v,v) + torch.exp(self.logsigman2)
        return pred_mean, pred_var


if __name__ == "__main__":

    learning_rate = 1 # 1 for BFGS, 0.001 for Adam
    no_iters = 200
    no_inputs = 1 # dimensionality of input
    BFGS = True # use Adam or BFGS 

    # load the 1D dataset
    data_location = '..//data//1D//1D_200.pkl'
    with open(data_location, 'rb') as f:
        train_inputs, train_outputs, test_inputs = pickle.load(f)
    no_train = train_outputs.size
    no_test = test_inputs.shape[0]

    # convert to torch tensors
    train_inputs = torch.Tensor(train_inputs)
    train_inputs = torch.unsqueeze(train_inputs, 1) # 1 dimensional data 
    train_outputs = torch.Tensor(train_outputs)
    test_inputs = torch.Tensor(test_inputs)
    test_inputs = torch.unsqueeze(test_inputs, 1) # 1 dimensional data

    # initialise model
    model = GP(no_inputs)

    # optimize hyperparameters
    if BFGS == True:
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with trange(no_iters) as t:
        for i in t:
            if BFGS == True:
                def closure():
                    optimizer.zero_grad()
                    NLL = -model.get_LL(train_inputs, train_outputs)
                    NLL.backward()
                    return NLL
                optimizer.step(closure)
                NLL = -model.get_LL(train_inputs, train_outputs)
            else: 
                optimizer.zero_grad()
                NLL = -model.get_LL(train_inputs, train_outputs)
                NLL.backward()
                optimizer.step() 
            # update tqdm 
            if i % 10 == 0:
                t.set_postfix(loss=NLL.item())

    # get posterior predictive
    pred_mean, pred_var = model.posterior_predictive(train_inputs, train_outputs, test_inputs)    

    # plot 1D regression
    fig, ax = plt.subplots()
    plt.plot(torch.squeeze(train_inputs).data.numpy(), train_outputs.data.numpy(), '+k')
    pred_mean = pred_mean.data.numpy()
    pred_var = pred_var.data.numpy()
    pred_sd = np.sqrt(pred_var)
    plt.plot(torch.squeeze(test_inputs).data.numpy(), pred_mean, color='b')
    plt.fill_between(torch.squeeze(test_inputs).data.numpy(), pred_mean + 2*pred_sd, 
            pred_mean - 2*pred_sd, color='b', alpha=0.3)
    filepath = 'full_GP_200.pdf'
    fig.savefig(filepath)
    plt.close()

    # print final NLL and hyperparameters
    file = open('results_200.txt','w') 
    file.write('1D GP regression with 20 points \n')
    file.write('noise_sd: {} \n'.format(torch.exp(model.logsigman2/2).item()))
    file.write('function_sd: {} \n'.format(torch.exp(model.logsigmaf2/2).item()))
    file.write('length_scale: {} \n'.format(torch.exp(model.logl2/2).item()))
    file.write('log_marginal_likelihood: {} \n'.format(model.get_LL(train_inputs, train_outputs).item()))
    file.close() 
