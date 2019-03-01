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
from variational_GP_no_Matern_proper_freezing import *
#from variational_GP import *
from full_GP_more_jitter import *

KL_jitter_factor = 10e-5

def gaussian_KL(mu0, mu1, Sigma0, Sigma1):
    # calculate the KL divergence between multivariate Gaussians KL(0||1)py
    no_dims = Sigma0.shape[0]

    L1 = torch.potrf(Sigma1 + KL_jitter_factor*torch.eye(Sigma1.shape[0]), upper=False)
    L1slashSigma = torch.trtrs(Sigma0 ,L1 ,upper=False)[0]
    SigmainvSigma = torch.trtrs(L1slashSigma ,L1.transpose(0,1))[0]
    trace_term = torch.trace(SigmainvSigma)

    mu_diff = mu1 - mu0
    v = torch.trtrs(mu_diff, L1, upper=False)[0]
    quadratic_term = v.transpose(0,1) @ v

    L0 = torch.potrf(Sigma0 + KL_jitter_factor*torch.eye(Sigma0.shape[0]), upper=False)
    logdet_term = 2*torch.sum(torch.log(torch.diag(L1))) - 2*torch.sum(torch.log(torch.diag(L0)))

    KL = 0.5*(trace_term + quadratic_term - no_dims + logdet_term)

    #import pdb; pdb.set_trace()

    return KL

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


    ### FULL GP ###

    # hyperparameters
    no_inputs = 13
    BFGS = False
    learning_rate = 0.01
    no_iters = 500

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
    full_pred_mean, full_pred_covar = model.joint_posterior_predictive(x_train_normalised, y_train_normalised, x_test_normalised, noise=True)

    # unnormalise the predictive outputs
    full_pred_mean = full_pred_mean*train_sd[-1] + train_mean[-1]
    full_pred_covar = full_pred_covar*(train_sd[-1])**2

    full_logsigmaf2 = deepcopy(model.logsigmaf2)
    full_logl2 = deepcopy(model.logl2)
    full_logsigman2 = deepcopy(model.logsigman2)

    ### VARIATIONAL GP ###

    grid = [1,15] + list(range(30,451,30)) + [455]
    no_replicates = 10
    KL_results = np.zeros([len(grid),no_replicates])

    for grid_index,no_inducing in enumerate(grid):
        for replicate_index in range(no_replicates):
            
            varGP = variational_GP(x_train_normalised.data.numpy(), np.expand_dims(y_train_normalised.data.numpy(),1), no_inducing=no_inducing, freeze_hyperparam=True, logsigmaf2=full_logsigmaf2, logl2=full_logl2, logsigman2=full_logsigman2)

            varGP.optimize_parameters(500, 'Adam', learning_rate=0.01)
            var_pred_mean, var_pred_covar = varGP.joint_posterior_predictive(x_test_normalised.data.numpy(), noise=True)
            var_pred_mean = var_pred_mean*train_sd[-1] + train_mean[-1]
            var_pred_covar = var_pred_covar*(train_sd[-1]**2)
            var_pred_var = torch.diag(var_pred_covar)
            var_pred_var = var_pred_var*(train_sd[-1])**2


            # Compute KL divergence
            KL = gaussian_KL(full_pred_mean, var_pred_mean, full_pred_covar, var_pred_covar)
            KL_results[grid_index,replicate_index] = KL
    
    np.savetxt('KL_results.tsv', KL_results, delimiter='\t')