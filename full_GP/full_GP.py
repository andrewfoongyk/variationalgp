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
    def __init__(self, no_inputs, kernel='SE'):
        super(GP, self).__init__()
        self.kernel = kernel

        # initialise hyperparameters
        self.no_inputs = no_inputs # input dimension
        self.logsigmaf2 = nn.Parameter(torch.Tensor([0])) # function variance
        self.logl2 = nn.Parameter(torch.zeros(no_inputs)) # horizontal length scales
        self.logsigman2 = nn.Parameter(torch.Tensor([0])) # noise variance
        self.jitter_factor = 1e-6

    def get_LL(self, train_inputs, train_outputs):
        # form the kernel matrix Knn using squared exponential ARD kernel
        train_inputs_col = torch.unsqueeze(train_inputs.transpose(0,1), 2)
        train_inputs_row = torch.unsqueeze(train_inputs.transpose(0,1), 1)
        squared_distances = (train_inputs_col - train_inputs_row)**2        
        length_factors = (1/(2*torch.exp(self.logl2))).reshape(self.no_inputs,1,1)        
        Knn = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))

        # cholesky decompose
        L = torch.potrf(Knn + torch.exp(self.logsigman2)*torch.eye(train_inputs.shape[0]) + self.jitter_factor*torch.eye(train_inputs.shape[0]), upper=False) # lower triangular decomposition
        Lslashy = torch.trtrs(train_outputs, L, upper=False)[0]
        alpha = torch.trtrs(Lslashy, torch.transpose(L,0,1))[0]

        # get log marginal likelihood
        LL = -0.5*torch.dot(train_outputs, torch.squeeze(alpha)) - torch.sum(torch.log(torch.diag(L))) - (train_inputs.shape[0]/2)*torch.log(torch.Tensor([2*3.1415926536]))
        return LL

    def get_K(self, input1, input2):
        if self.kernel == 'SE':
            # form the kernel matrix with dimensions (input1 x input2) using squared exponential ARD kernel
            inputs_col = torch.unsqueeze(input1.transpose(0,1), 2)
            inputs_row = torch.unsqueeze(input2.transpose(0,1), 1)
            squared_distances = (inputs_col - inputs_row)**2        
            length_factors = (1/(2*torch.exp(self.logl2))).reshape(self.no_inputs,1,1)        
            K = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))
        elif self.kernel == 'matern':
            inputs_col = torch.unsqueeze(input1.transpose(0,1), 2)
            inputs_row = torch.unsqueeze(input2.transpose(0,1), 1)
            abs_distances = torch.abs(inputs_col - inputs_row)
            length_factors = (1/(torch.exp(self.logl2))).reshape(self.no_inputs,1,1)
            scaled_distances = abs_distances * length_factors
            K = torch.exp(self.logsigmaf2) * torch.exp(-np.sqrt(3)*torch.sum(scaled_distances, 0) + torch.sum(torch.log(1 + np.sqrt(3)*scaled_distances), 0))
        else:
            raise Exception('Invalid kernel name')
        return K

    def posterior_predictive(self, train_inputs, train_outputs, test_inputs):
        # get covariance matrix
        Knn = self.get_K(train_inputs, train_inputs)
        train_inputs_col = torch.unsqueeze(train_inputs.transpose(0,1), 2)
        
        no_test = test_inputs.shape[0]
        pred_mean = torch.zeros(no_test)
        pred_var = torch.zeros(no_test)

        # cholesky decompose
        L = torch.potrf(Knn + torch.exp(self.logsigman2)*torch.eye(train_inputs.shape[0]) + self.jitter_factor*torch.eye(train_inputs.shape[0]), upper=False) # lower triangular decomposition
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

            # get predictive variance with noise variance added
            v = torch.squeeze(torch.trtrs(kstar, L, upper=False)[0])
            pred_var[i] = torch.exp(self.logsigmaf2) - torch.dot(v,v) + torch.exp(self.logsigman2)
        return pred_mean, pred_var

    def joint_posterior_predictive(self, train_inputs, train_outputs, test_inputs, noise=False):
        # return the joint posterior, which is a multivariate Gaussian
        no_test = test_inputs.shape[0]
        # get training inputs covariance matrix
        Knn = self.get_K(train_inputs, train_inputs)
        
        # cholesky decompose
        L = torch.potrf(Knn + torch.exp(self.logsigman2)*torch.eye(train_inputs.shape[0]) + self.jitter_factor*torch.eye(train_inputs.shape[0]), upper=False) # lower triangular decomposition
        Lslashy = torch.trtrs(train_outputs, L, upper=False)[0]
        
        # get cross covariance between test and train points, Ktn
        Ktn = self.get_K(test_inputs, train_inputs)
        Knt = Ktn.transpose(0,1)

        # get predictive mean 
        LslashKnt = torch.trtrs(Knt, L, upper=False)[0]
        pred_mean = LslashKnt.transpose(0,1) @ Lslashy
 
        # get predictive covariance
        Ktt = self.get_K(test_inputs, test_inputs)
        if noise == True: # add observation noise
            pred_cov = Ktt + torch.exp(self.logsigman2)*torch.eye(no_test) - LslashKnt.transpose(0,1) @ LslashKnt
        else:
            pred_cov = Ktt - LslashKnt.transpose(0,1) @ LslashKnt + 1e-6*torch.eye(no_test)

        return pred_mean, pred_cov

def gaussian_KL(mu0, mu1, Sigma0, Sigma1):
    # calculate the KL divergence between multivariate Gaussians KL(0||1)
    no_dims = Sigma0.shape[0]

    L1 = torch.potrf(Sigma1 + self.jitter_factor*torch.eye(train_inputs.shape[0]), upper=False)
    L1slashSigma = torch.trtrs(Sigma0 ,L1 ,upper=False)[0]
    SigmainvSigma = torch.trtrs(L1slashSigma ,L1.transpose(0,1))[0]
    trace_term = torch.trace(SigmainvSigma)

    mu_diff = mu1 - mu0
    v = torch.trtrs(mu_diff, L1, upper=False)[0]
    quadratic_term = v.transpose(0,1) @ v

    L0 = torch.potrf(Sigma0 + self.jitter_factor*torch.eye(train_inputs.shape[0]), upper=False)
    logdet_term = 2*torch.sum(torch.log(torch.diag(L1))) - 2*torch.sum(torch.log(torch.diag(L0)))

    KL = 0.5*(trace_term + quadratic_term - no_dims + logdet_term)
    return KL

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    learning_rate = 1 # 1 for BFGS, 0.001 for Adam
    no_iters = 200
    no_inputs = 1 # dimensionality of input
    BFGS = True # use Adam or BFGS 

    # load the 1D dataset
    data_location = '..//data//1D//1D_20.pkl'
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
    filepath = 'full_GP_20.pdf'
    fig.savefig(filepath)
    plt.close()

    # plot samples from the posterior using the full predictive distribution
    pred_mean, pred_cov = model.joint_posterior_predictive(train_inputs, train_outputs, test_inputs, noise=False)
    pred_mean = torch.squeeze(pred_mean)
    pred_mean = pred_mean.data.numpy()
    pred_cov = pred_cov.data.numpy()

    no_samples = 10
    # sample from the posterior predictive
    samples = np.random.multivariate_normal(pred_mean, pred_cov, no_samples)
    # plot the samples
    fig, ax = plt.subplots()
    plt.plot(torch.squeeze(train_inputs).data.numpy(), train_outputs.data.numpy(), '+k')
    for i in range(no_samples):
        plt.plot(torch.squeeze(test_inputs).data.numpy(), samples[i,:])
    filepath = 'full_GP_20_samples.pdf'
    fig.savefig(filepath)
    plt.close()

    # print final NLL and hyperparameters
    file = open('results_20.txt','w') 
    file.write('1D GP regression with 20 points \n')
    file.write('noise_sd: {} \n'.format(torch.exp(model.logsigman2/2).item()))
    file.write('function_sd: {} \n'.format(torch.exp(model.logsigmaf2/2).item()))
    file.write('length_scale: {} \n'.format(torch.exp(model.logl2/2).item()))
    file.write('log_marginal_likelihood: {} \n'.format(model.get_LL(train_inputs, train_outputs).item()))
    file.close() 
