# generate a 1D dataset using the Matern kernel

import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.append('..//..//full_GP')
from full_GP import GP

# set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

no_train = 200
no_test = 301
noise_variance = 0.2**2

train_inputs = np.random.uniform(0,6,no_train)
train_inputs_tensor = torch.Tensor(train_inputs).unsqueeze(1)

test_inputs = np.linspace(-3, 10, no_test)

myGP = GP(1, kernel='matern')
cov = myGP.get_K(train_inputs_tensor, train_inputs_tensor).data.numpy()
cov = cov + noise_variance*np.eye(no_train)
samples = np.random.multivariate_normal(np.zeros(no_train), cov)

train_outputs = samples

fig, ax = plt.subplots()
plt.scatter(train_inputs, train_outputs)
filepath = '1D_dataset_200_matern.pdf'
fig.savefig(filepath)
plt.close()

# create smaller dataset with only 20 points
train_inputs_20 = train_inputs[0::10]
train_outputs_20 = train_outputs[0::10]
fig, ax = plt.subplots()
plt.scatter(train_inputs_20, train_outputs_20)
filepath = '1D_dataset_20_matern.pdf'
fig.savefig(filepath)
plt.close()

# pickle the full and subsampled datasets
filename = '1D_200_matern.pkl'
# pickle the dataset 
with open(filename, 'wb') as f:
    pickle.dump([train_inputs, train_outputs, test_inputs], f)

filename = '1D_20_matern.pkl'
# pickle the dataset 
with open(filename, 'wb') as f:
    pickle.dump([train_inputs_20, train_outputs_20, test_inputs], f)
