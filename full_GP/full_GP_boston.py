# train a full GP on the Boston Housing dataset

from full_GP import GP
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # unpickle the boston housing dataset
    data_location = '..//data//boston_housing//boston_housing0.pkl'
    with open(data_location, 'rb') as f:
        train_set, train_set_normalised, val_set_normalised, test_set, train_mean, train_sd = pickle.load(f)
    
    train_mean = torch.Tensor(train_mean)
    train_sd = torch.Tensor(train_sd)

    x_train_normalised = torch.Tensor(train_set_normalised[:,:-1])
    y_train_normalised = torch.Tensor(train_set_normalised[:,-1])

    x_val_normalised = torch.Tensor(val_set_normalised[:,:-1])
    y_val_normalised = torch.Tensor(val_set_normalised[:,-1])

    # combine train and val sets
    x_train_normalised = torch.cat((x_train_normalised, x_val_normalised), 0)
    y_train_normalised = torch.cat((y_train_normalised, y_val_normalised), 0)

    x_test = torch.Tensor(test_set[:,:-1])
    y_test = torch.Tensor(test_set[:,-1])

    # normalise the test inputs
    x_test_normalised = x_test - train_mean[:-1]
    x_test_normalised = x_test_normalised/train_sd[:-1]
    
    # hyperparameters
    no_inputs = 13
    BFGS = True
    learning_rate = 1
    no_iters = 100

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
                    NLL = -model.get_LL(x_train_normalised, y_train_normalised)
                    NLL.backward()
                    return NLL
                optimizer.step(closure)
                NLL = -model.get_LL(x_train_normalised, y_train_normalised)
            else: 
                optimizer.zero_grad()
                NLL = -model.get_LL(x_train_normalised, y_train_normalised)
                NLL.backward()
                optimizer.step() 
            # update tqdm 
            if i % 10 == 0:
                t.set_postfix(loss=NLL.item())

    # get posterior predictive
    pred_mean, pred_var = model.posterior_predictive(x_train_normalised, y_train_normalised, x_test_normalised)
    # unnormalise the predictive outputs
    pred_mean = pred_mean*train_sd[-1] + train_mean[-1]
    pred_var = pred_var*(train_sd[-1])**2

    # get the standardized mean square error "SMSE"
    SMSE = torch.mean((pred_mean - y_test)**2)/torch.var(y_test)

    # get the standardized negative log probability density "SNLP" - following Section 2.5 in Rasmussen and Williams
    # get test negative LL
    NLL = 0.5*torch.sum(torch.log(2*3.1415926536*pred_var)) + torch.sum((pred_mean - y_test)**2/(2*pred_var)) 
    # get trivial model negative LL
    no_test = y_test.shape[0]
    NLL_trivial = (no_test/2)*torch.log(2*3.1415926536*train_sd[-1]**2) + torch.sum((y_test - train_mean[-1])**2/(2*train_sd[-1]**2))
    SLL = NLL - NLL_trivial
    MSLL = SLL/no_test

    # print final log marginal likelihood, SMSE, SNLP(MSLL) 
    file = open('results_boston.txt','w') 
    file.write('Full GP regression on Boston Housing \n')
    file.write('noise_sd: {} \n'.format(torch.exp(model.logsigman2/2).item()))
    file.write('function_sd: {} \n'.format(torch.exp(model.logsigmaf2/2).item()))
    file.write('length_scales: {} \n'.format(torch.exp(model.logl2/2).data.numpy()))
    file.write('log_marginal_likelihood: {} \n'.format(model.get_LL(x_train_normalised, y_train_normalised).item()))
    file.write('SMSE: {} \n'.format(SMSE.item()))
    file.write('SNLP: {} \n'.format(MSLL.item()))
    file.close() 

    