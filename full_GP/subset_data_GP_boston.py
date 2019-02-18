# train full GP subsets of the data of Boston Housing. Compare the results with the full GP as the number of 'inducing points' increases

from full_GP import GP
from full_GP import gaussian_KL
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt

def train(x_train_normalised, y_train_normalised, x_test, no_inputs=13, BFGS=True, learning_rate=1, no_iters=200, return_hypers=False):
    # train the GP and return the joint predictive distribution on the test inputs
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

    # get joint posterior predictive of the latent function values
    pred_mean, pred_cov = model.joint_posterior_predictive(x_train_normalised, y_train_normalised, x_test, noise=False)
    if return_hypers == True:
        logl2 = model.logl2
        logsigmaf2 = model.logsigmaf2
        logsigman2 = model.logsigman2
        return pred_mean, pred_cov, logl2, logsigmaf2, logsigman2
    else:
        return pred_mean, pred_cov

def subset_data_predict(model, subsample_list, x_train_normalised, y_train_normalised, x_test_normalised, pred_mean_full, pred_cov_full):
    # get the KL divergence to the true posterior as a function of subsampling
    KL_list = np.zeros(len(subsample_list))
    no_train = x_train_normalised.shape[0]
    for i, no_points in enumerate(subsample_list):
        # pick a random subset of the data
        ind = np.random.choice(no_train, no_points, replace=False)
        x_train_normalised_subsampled = x_train_normalised[ind,:]
        y_train_normalised_subsampled = y_train_normalised[ind]

        # get joint posterior predictive with subset of data
        pred_mean, pred_cov = model.joint_posterior_predictive(x_train_normalised_subsampled, y_train_normalised_subsampled, x_test_normalised, noise=False)
        
        # calculate KL divergence between subsampled posterior and full posterior
        KL = gaussian_KL(pred_mean_full, pred_mean, pred_cov_full, pred_cov)
        KL_list[i] = KL.data.numpy()
    return KL_list

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    directory = 'experiments//subset_data//'

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

    # train on the full data set, and keep the hyperparameters learned
    pred_mean_full, pred_cov_full, logl2, logsigmaf2, logsigman2 = train(x_train_normalised, y_train_normalised, x_test_normalised, no_inputs, BFGS, learning_rate, no_iters, return_hypers=True)

    # subsample the training set and do inference
    subsample_list = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 455]    
    # initialise model using full GP hyperparameters
    model = GP(no_inputs)
    model.logl2 = logl2
    model.logsigmaf2 = logsigmaf2
    model.logsigman2 = logsigman2

    no_exps = 10
    KL_lists = np.zeros((no_exps, len(subsample_list))) 
    for i in range(no_exps): # repeat the experiment 10 times
        KL_lists[i,:] = subset_data_predict(model, subsample_list, x_train_normalised, y_train_normalised, x_test_normalised, pred_mean_full, pred_cov_full)
    KL_mean = np.mean(KL_lists, axis=0)
    KL_se = np.std(KL_lists, axis=0)/np.sqrt(no_exps) 
    # ^ standard error of the mean - can't tell if Titsias uses standard deviation or 
    # standard error in the paper - he says standard error but plot looks more like standard deviation

    # plot the KL against number of subsampled points
    fig, ax = plt.subplots()
    plt.errorbar(np.array(subsample_list), KL_mean, yerr = KL_se, capsize=5)
    plt.xlabel('Number of inducing variables')
    plt.ylabel('KL(p||q)')
    filepath = directory + 'subset_data_KL.pdf'
    fig.savefig(filepath)
    plt.close()
        

