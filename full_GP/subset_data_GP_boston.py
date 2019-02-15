# train a full GP on a subset of the data of Boston Housing. Compare the results with the full GP as the number of 'inducing points' increases

from full_GP import GP
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt

def subset_data_train(no_points, x_train_normalised, y_train_normalised):
    # train the GP on a subset of the data and return the joint predictive distribution on the test inputs
    


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