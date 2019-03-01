import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from tqdm import trange
from copy import deepcopy

from error_functions import *
sys.path.append('/home/frozenmiwe/7_MGM/Advanced ML/code/full_GP')
sys.path.append('/home/frozenmiwe/7_MGM/Advanced ML/code/variational_GP')
from variational_GP import *
from full_GP_more_jitter import *

if __name__ == '__main__':

    # Load the Boston dataset
    data_location = '../data/boston_housing/boston_housing0.pkl'
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

    '''
    ### FULL GP ###

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
                NLL = -model.get_LL(x_train_normalised, y_train_normalised)logl2
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

    SMSE_fullGP = SMSE(pred_mean, y_test)
    SNLP_fullGP = SNLP(pred_mean, pred_var, y_test, train_mean[-1], train_sd[-1])

    #import pdb; pdb.set_trace()
    print(SMSE_fullGP,SNinpuLP_fullGP)
    '''

    ### VARIATIONAL GP ###

    #Set random seed for reproducibility
    random.seed(0)

    grid = [1,15] + list(range(30,451,30)) + [455]
    no_replicates = 10
    SMSE_results = np.zeros([len(grid),no_replicates])
    SNLP_results = np.zeros([len(grid),no_replicates])

    for grid_index,no_inducing in enumerate(grid):
        for replicate_index in range(no_replicates):

            varGP = variational_GP(x_train_normalised.data.numpy(), np.expand_dims(y_train_normalised.data.numpy(),1), no_inducing=no_inducing)
            varGP.optimize_parameters(1000, 'Adam', learning_rate=0.01)
            pred_mean, pred_covar = varGP.joint_posterior_predictive(x_test_normalised.data.numpy(), noise=True)
            pred_mean = torch.squeeze(pred_mean*train_sd[-1] + train_mean[-1])
            pred_var = torch.diag(pred_covar)
            pred_var = pred_var*(train_sd[-1])**2

            SNLP_varGP = SNLP(pred_mean, pred_var, y_test, train_mean[-1], train_sd[-1])
            SMSE_varGP = SMSE(torch.Tensor(pred_mean), y_test)

            SMSE_results[grid_index,replicate_index] = SMSE_varGP
            SNLP_results[grid_index,replicate_index] = SNLP_varGP

    np.savetxt('SMSE_results.tsv', SMSE_results, delimiter='\t')
    np.savetxt('SNLP_results.tsv', SNLP_results, delimiter='\t')        
