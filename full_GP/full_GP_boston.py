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
    # unpickle the boston housing dataset
    data_location = '..//data//boston_housing//boston_housing0.pkl'
    with open(data_location, 'rb') as f:
        train_set, train_set_normalised, val_set_normalised, test_set, train_mean, train_sd = pickle.load(f)
    
    train_mean = torch.Tensor(train_mean).cuda()
    train_sd = torch.Tensor(train_sd).cuda()

    x_train_normalised = torch.Tensor(train_set_normalised[:,:-1]).cuda()
    y_train_normalised = torch.Tensor(train_set_normalised[:,-1]).cuda()

    x_val_normalised = torch.Tensor(val_set_normalised[:,:-1]).cuda()
    y_val_normalised = torch.Tensor(val_set_normalised[:,-1]).cuda()

    # combine train and val sets
    x_train_normalised = torch.cat((x_train_normalised, x_val_normalised), 0)
    y_train_normalised = torch.cat((y_train_normalised, y_val_normalised), 0)

    x_test = torch.Tensor(test_set[:,:-1]).cuda()
    y_test = torch.Tensor(test_set[:,-1]).cuda()
    
