# unpack the 1D dataset from Snelson and Gharahmani and pickle it

import numpy as np
import pickle
import matplotlib.pyplot as plt

test_inputs = np.loadtxt('test_inputs')
train_inputs = np.loadtxt('train_inputs')
train_outputs = np.loadtxt('train_outputs')

fig, ax = plt.subplots()
plt.scatter(train_inputs, train_outputs)
filepath = '1D_dataset_200.pdf'
fig.savefig(filepath)
plt.close()

# create smaller dataset with only 20 points
train_inputs_20 = train_inputs[0::10]
train_outputs_20 = train_outputs[0::10]
fig, ax = plt.subplots()
plt.scatter(train_inputs_20, train_outputs_20)
filepath = '1D_dataset_20.pdf'
fig.savefig(filepath)
plt.close()

# pickle the full and subsampled datasets
filename = '1D_200.pkl'
# pickle the dataset 
with open(filename, 'wb') as f:
    pickle.dump([train_inputs, train_outputs, test_inputs], f)

filename = '1D_20.pkl'
# pickle the dataset 
with open(filename, 'wb') as f:
    pickle.dump([train_inputs_20, train_outputs_20, test_inputs], f)


