# plot Fig 1 column 1 with Matern kernel

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
sys.path.append('..//full_GP')
sys.path.append('..//variational_GP')
from variational_GP import variational_GP
from full_GP import GP

if __name__ == "__main__": 
	# set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    results_directory = '..//figures//figure1_column1_matern//'

	# load the 1D dataset
    with open('../data/1D/1D_200_matern.pkl', 'rb') as f:
        train_inputs, train_outputs, test_inputs = pickle.load(f)
    no_train = train_outputs.size
    no_test = test_inputs.shape[0]

    # convert to torch tensors
    train_inputs = torch.Tensor(train_inputs)
    train_inputs = torch.unsqueeze(train_inputs, 1) # 1 dimensional data 
    train_outputs = torch.Tensor(train_outputs)
    test_inputs = torch.Tensor(test_inputs)
    test_inputs = torch.unsqueeze(test_inputs, 1) # 1 dimensional data

    ### VARIATIONAL GP ###

    no_inducing = 15
    varGP = variational_GP(train_inputs.data.numpy(), np.expand_dims(train_outputs.data.numpy(),1), no_inducing=no_inducing, kernel='matern')
    var_pred_mean, var_pred_covar = varGP.joint_posterior_predictive(test_inputs.data.numpy())

    # record initial inducing point locations
    initial_inducing = deepcopy(torch.squeeze(varGP.Xm).data.numpy())

    varGP.optimize_parameters(10000, 'Adam', learning_rate=0.001)
    var_pred_mean, var_pred_covar = varGP.joint_posterior_predictive(test_inputs.data.numpy(), noise=True) # plot error bars with observation noise

    # final inducing points
    final_inducing = torch.squeeze(varGP.Xm).data.numpy()

    # compute LL

    var_pred_mean = (torch.squeeze(var_pred_mean)).data.numpy()
    var_pred_var = (torch.diag(var_pred_covar)).data.numpy()
    var_pred_sd = np.sqrt(var_pred_var)


    ### FULL GP ###

    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    learning_rate = 1 # 1 for BFGS, 0.001 for Adam
    no_iters = 500
    no_inputs = 1 # dimensionality of input
    BFGS = True # use Adam or BFGS 

    # initialise fullGP
    fullGP = GP(no_inputs, kernel='matern')

    # optimize hyperparameters
    if BFGS == True:
        optimizer = optim.LBFGS(fullGP.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(fullGP.parameters(), lr=learning_rate)

    with trange(no_iters) as t:
        for i in t:
            if BFGS == True:
                def closure():
                    optimizer.zero_grad()
                    NLL = -fullGP.get_LL(train_inputs, train_outputs)
                    NLL.backward()
                    return NLL
                optimizer.step(closure)
                NLL = -fullGP.get_LL(train_inputs, train_outputs)
            else: 
                optimizer.zero_grad()
                NLL = -fullGP.get_LL(train_inputs, train_outputs)
                NLL.backward()
                optimizer.step() 
            # update tqdm 
            if i % 10 == 0:
                t.set_postfix(loss=NLL.item())


    # plot samples from the posterior using the full predictive distribution
    full_pred_mean, full_pred_covar = fullGP.joint_posterior_predictive(train_inputs, train_outputs, test_inputs, noise=True)
    full_pred_mean = torch.squeeze(full_pred_mean)
    full_pred_mean = full_pred_mean.data.numpy()
    full_pred_var = (torch.diag(full_pred_covar)).data.numpy()
    full_pred_sd = np.sqrt(full_pred_var)

    # record full hyperparams and log marginal likelihood
    file = open(results_directory + 'results_200_matern.txt','w') 
    file.write('variational GP results: \n')
    file.write('noise_sd: {} \n'.format(torch.exp(varGP.logsigman2/2).item()))
    file.write('function_sd: {} \n'.format(torch.exp(varGP.logsigmaf2/2).item()))
    file.write('length_scale: {} \n'.format(torch.exp(varGP.logl2/2).item()))
    file.write('variational free energy Fv: {} \n'.format(varGP.Fv().item()))
    file.write(' \n')
    file.write('full GP results: \n')
    file.write('noise_sd: {} \n'.format(torch.exp(fullGP.logsigman2/2).item()))
    file.write('function_sd: {} \n'.format(torch.exp(fullGP.logsigmaf2/2).item()))
    file.write('length_scale: {} \n'.format(torch.exp(fullGP.logl2/2).item()))
    file.write('log_marginal_likelihood: {} \n'.format(fullGP.get_LL(train_inputs, train_outputs).item()))
    file.close() 


    ### PLOT ###

    fig, ax = plt.subplots()

    # plot inducing point locations
    plt.scatter(initial_inducing, 2*np.ones(no_inducing), s=50, marker = '+')
    plt.scatter(final_inducing, -3*np.ones(no_inducing), s=50, marker = '+')

    # plot varGP predictions
    plt.plot(torch.squeeze(train_inputs).data.numpy(), train_outputs.data.numpy(), '+k')
    plt.plot(torch.squeeze(test_inputs).data.numpy(), var_pred_mean, color='b')
    plt.plot(torch.squeeze(test_inputs).data.numpy(), var_pred_mean + 2*var_pred_sd, color='b')
    plt.plot(torch.squeeze(test_inputs).data.numpy(), var_pred_mean - 2*var_pred_sd, color='b')

    # plot fullGP predictions

    plt.plot(torch.squeeze(test_inputs).data.numpy(), full_pred_mean, color='r', linestyle='--')
    plt.plot(torch.squeeze(test_inputs).data.numpy(), full_pred_mean + 2*full_pred_sd, color='r', linestyle='--')
    plt.plot(torch.squeeze(test_inputs).data.numpy(), full_pred_mean - 2*full_pred_sd, color='r', linestyle='--')

    # save plot
    filepath = results_directory + 'var_vs_fullGP_200_matern.pdf'
    fig.savefig(filepath)
    plt.close()