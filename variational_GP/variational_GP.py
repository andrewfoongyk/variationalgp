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

class variational_GP(nn.Module):  
    def __init__(self, Xn, Yn, no_inducing=15, kernel='SE'):   # the GP takes the training data as arguments, in the form of numpy arrays, with the correct dimensions        
        super(variational_GP, self).__init__()
        self.kernel = kernel

        # initialise hyperparameters
        self.Xn = torch.tensor(Xn).type(torch.FloatTensor)
        self.Yn = torch.tensor(Yn).type(torch.FloatTensor)
        self.no_inputs = Xn.shape[1]
        self.logsigmaf2 = nn.Parameter(torch.Tensor([0])) # function variance
        self.logl2 = nn.Parameter(torch.zeros(self.no_inputs)) # horizontal length scales
        self.logsigman2 = nn.Parameter(torch.Tensor([0])) # noise variance
        self.jitter_factor = 1e-6

        # initialise inducing points randomly
        #import pdb; pdb.set_trace()
        upper = 3
        lower = 2
        input_locations = (lower - upper) * torch.rand(no_inducing,self.Xn.shape[1]) + upper
        self.Xm = nn.Parameter(input_locations.type(torch.FloatTensor),requires_grad=True)
        #self.Xm = nn.Parameter(torch.Tensor(Xn[random.sample(range(Xn.shape[0]),no_inducing),:]).type(torch.FloatTensor))
        
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
            K = torch.exp(self.logsigmaf2) * torch.exp(-1.732051*torch.sum(scaled_distances, 0) + torch.sum(torch.log(1 + 1.732051*scaled_distances), 0))
        else:
            raise Exception('Invalid kernel name')
        return K
    
    def Fv(self): # All the necessary arguments are instance variables, so no need to pass them
        no_train = self.Xn.shape[0]
        no_inducing = self.Xm.shape[0]

        # Calculate kernel matrices 
        Kmm = self.get_K(self.Xm, self.Xm)
        Knm = self.get_K(self.Xn, self.Xm)
        Kmn = Knm.transpose(0,1)

        # calculate the 'inner matrix' and Cholesky decompose
        M = Kmm + torch.exp(-self.logsigman2) * Kmn @ Knm
        L = torch.potrf(M + torch.mean(torch.diag(M))*self.jitter_factor*torch.eye(no_inducing), upper=False)

        # Compute first term (log of Gaussian pdf)
        # constant term
        constant_term = -(no_train/2) * torch.log(torch.Tensor([2*np.pi]))
        
        # quadratic term - Yn should be a column vector
        LslashKmny = torch.trtrs(Kmn @ self.Yn, L, upper=False)[0]
        quadratic_term = -0.5 * (torch.exp(-self.logsigman2) * self.Yn.transpose(0,1) @ self.Yn - torch.exp(-2*self.logsigman2) * LslashKmny.transpose(0,1) @ LslashKmny )

        # logdet term
        # Cholesky decompose the Kmm
        L_inducing = torch.potrf(Kmm + torch.mean(torch.diag(Kmm))*self.jitter_factor*torch.eye(no_inducing), upper=False)
        logdet_term = -0.5 * ( 2*torch.sum(torch.log(torch.diag(L))) - 2*torch.sum(torch.log(torch.diag(L_inducing))) + no_train * self.logsigman2 )

        log_gaussian_term = constant_term + logdet_term + quadratic_term

        # Compute the second term (trace regulariser)
        B = torch.trtrs(Kmn , L_inducing, upper=False)[0]
        trace_term = -0.5 * torch.exp(-self.logsigman2) * ( no_train*torch.exp(self.logsigmaf2) - torch.sum(B**2) )

        return log_gaussian_term + trace_term

    def joint_posterior_predictive(self, test_inputs, noise=False): # assume test_inputs is a numpy array
        # get the mean and covariance of the joint Gaussian posterior over the test outputs
        test_inputs = torch.Tensor(test_inputs).type(torch.FloatTensor) 
        no_test = test_inputs.shape[0]
        no_inducing = self.Xm.shape[0]   
       
        # Calculate kernel matrices 
        Kxx = self.get_K(test_inputs, test_inputs)
        Kmx = self.get_K(self.Xm, test_inputs)
        Kmm = self.get_K(self.Xm, self.Xm)
        Knm = self.get_K(self.Xn, self.Xm)
        Kmn = Knm.transpose(0,1)

        # calculate the 'inner matrix' and Cholesky decompose
        M = Kmm + torch.exp(-self.logsigman2) * Kmn @ Knm
        L = torch.potrf(M + torch.mean(torch.diag(M))*self.jitter_factor*torch.eye(no_inducing), upper=False)

        # Cholesky decompose the Kmm
        L_inducing = torch.potrf(Kmm + torch.mean(torch.diag(Kmm))*self.jitter_factor*torch.eye(no_inducing), upper=False)

        # backsolve 
        LindslashKmx = torch.trtrs(Kmx, L_inducing, upper=False)[0]
        LslashKmx = torch.trtrs(Kmx, L, upper=False)[0]

        cov = Kxx - LindslashKmx.transpose(0,1) @ LindslashKmx + LslashKmx.transpose(0,1) @ LslashKmx

        if noise == True: # add observation noise
            cov = cov + torch.exp(self.logsigman2) * torch.eye(no_test)

        # calculate the predictive mean by backsolving
        LslashKmny = torch.trtrs(Kmn @ self.Yn, L, upper=False)[0]

        mean = torch.exp(-self.logsigman2) * LslashKmx.transpose(0,1) @ LslashKmny

        return mean, cov            
    
    def optimize_parameters(self, no_iters, method = 'Adam', learning_rate=0.1):

        # optimize hyperparameters and inducing points
        if method == 'BFGS':
            optimizer = optim.LBFGS(self.parameters(), lr=learning_rate)
        elif method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            raise Exception('{} is not a valid method'.format(method))

        with trange(no_iters) as t:
            for i in t:
                if method == 'BFGS':
                    def closure():
                        optimizer.zero_grad()
                        negFv = -self.Fv()
                        negFv.backward()
                        return negFv
                    optimizer.step(closure)
                    negFv = -self.Fv()
                elif method == 'Adam': 
                    optimizer.zero_grad()
                    negFv = -self.Fv()
                    negFv.backward()
                    optimizer.step() 
                # update tqdm 
                if i % 10 == 0:
                    t.set_postfix(loss=negFv.item())
                #    print(-negFv)

if __name__ == "__main__": 
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

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

    no_inducing = 15
    myGP = variational_GP(train_inputs.data.numpy(), np.expand_dims(train_outputs.data.numpy(),1), no_inducing=no_inducing, kernel='matern')
    pred_mean, pred_covar = myGP.joint_posterior_predictive(test_inputs.data.numpy())

    # record initial inducing point locations
    initial_inducing = deepcopy(torch.squeeze(myGP.Xm).data.numpy())

    # plot and save
    fig, ax = plt.subplots()
    plt.plot(torch.squeeze(train_inputs).data.numpy(), train_outputs.data.numpy(), '+k')
    pred_mean = (torch.squeeze(pred_mean)).data.numpy()
    pred_var = (torch.diag(pred_covar)).data.numpy()
    pred_sd = np.sqrt(pred_var)
    plt.plot(torch.squeeze(test_inputs).data.numpy(), pred_mean, color='b')
    plt.fill_between(torch.squeeze(test_inputs).data.numpy(), pred_mean + 2*pred_sd, 
            pred_mean - 2*pred_sd, color='b', alpha=0.3)
    filepath = 'var_GP_200_matern.pdf'
    fig.savefig(filepath)
    plt.close()

    myGP.optimize_parameters(5000, 'Adam', learning_rate=0.01)
    pred_mean, pred_covar = myGP.joint_posterior_predictive(test_inputs.data.numpy(), noise=True) # plot error bars with observation noise

    # final inducing points
    final_inducing = torch.squeeze(myGP.Xm).data.numpy()

    # compute LL
    print(myGP.Fv())

    # plot after
    fig, ax = plt.subplots()
    # plot inducing point locations
    plt.scatter(initial_inducing, 2*np.ones(no_inducing), s=50, marker = '+')
    plt.scatter(final_inducing, -3*np.ones(no_inducing), s=50, marker = '+')

    plt.plot(torch.squeeze(train_inputs).data.numpy(), train_outputs.data.numpy(), '+k')
    pred_mean = (torch.squeeze(pred_mean)).data.numpy()
    pred_var = (torch.diag(pred_covar)).data.numpy()
    pred_sd = np.sqrt(pred_var)
    plt.plot(torch.squeeze(test_inputs).data.numpy(), pred_mean, color='b')
    plt.fill_between(torch.squeeze(test_inputs).data.numpy(), pred_mean + 2*pred_sd, 
            pred_mean - 2*pred_sd, color='b', alpha=0.3)
    filepath = 'var_GP_200_after_matern.pdf'
    fig.savefig(filepath)
    plt.close()