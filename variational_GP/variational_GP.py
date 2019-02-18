import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import random

class variational_GP(nn.Module):  
    def __init__(self, Xn, Yn):   # the GP takes the training data as arguments, in the form of numpy arrays, with the correct dimensions
        
        super().__init__()
        # initialise hyperparameters
        self.Xm = nn.Parameter(torch.Tensor(Xn[random.sample(range(Xn.shape[0]),15),:]).type(torch.FloatTensor))
        self.Xn = torch.tensor(Xn).type(torch.FloatTensor)
        self.Yn = torch.tensor(Yn).type(torch.FloatTensor)
        self.no_inputs = Xn.shape[1]
        self.logsigmaf2 = nn.Parameter(torch.Tensor([0])) # function variance
        self.logl2 = nn.Parameter(torch.zeros(self.no_inputs)) # horizontal length scales
        self.logsigman2 = nn.Parameter(torch.Tensor([0])) # noise variance
        self.jitter_factor = 1e-4
        self.log_marg = 0
        self.reg = 0
        
    def get_K(self,inputs1,inputs2):
        
        inputs1_col = torch.unsqueeze(inputs1.transpose(0,1), 2)
        inputs2_row = torch.unsqueeze(inputs2.transpose(0,1), 1)
        squared_distances = (inputs1_col - inputs2_row)**2        
        length_factors = (1/(2*torch.exp(self.logl2))).reshape(self.no_inputs,1,1)
        K = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))
        return(K)
    
    def Fv(self): # All the necessary arguments are instance variables, so no need to pass them
        no_train = self.Xn.shape[0]
        # Compute first term (log marginal likelihood)
        
        M = self.get_K(self.Xm,self.Xm) + 1/torch.exp(self.logsigman2) * torch.mm(self.get_K(self.Xm,self.Xn),self.get_K(self.Xn,self.Xm))
        M = M + torch.eye(M.shape[0])*self.jitter_factor
        L = torch.potrf(M,upper=False)
        LslashKmnYn, _ = torch.trtrs(torch.mm(self.get_K(self.Xm,self.Xn),self.Yn),L,upper=False)

        Kmm = self.get_K(self.Xm,self.Xm)
        
        term_pi = - self.Xn.shape[1] / 2 * torch.log(torch.Tensor([2*np.pi]))

        logdetM = 2*torch.sum(torch.log(torch.diag(L)))
        Lmm = torch.potrf(Kmm + 1e-6*torch.eye(Kmm.shape[0]), upper=False)
        logdetKmm = 2*torch.sum(torch.log(torch.diag(Lmm)))

        term_det = - 1 / 2 * (logdetM - logdetKmm +  no_train*self.logsigman2)
        term_quadratics =  - 1/2*(1/torch.exp(self.logsigman2)*torch.mm(self.Yn.transpose(0,1),self.Yn) 
                                   - 1/torch.exp(self.logsigman2)**2 * torch.mm(LslashKmnYn.transpose(0,1),LslashKmnYn))
        log_marg = term_pi + term_det + term_quadratics               
        
        # Compute second term (trace, regularizer)
        TrKnn = 0
        for elem in self.Xn:
            TrKnn += self.get_K(elem.unsqueeze(0),elem.unsqueeze(0))
        
        Kmm = self.get_K(self.Xm,self.Xm)
        Kmm = Kmm + torch.eye(Kmm.shape[0])*self.jitter_factor
        L = torch.potrf(Kmm,upper=False)
        LslashKmn, _ = torch.trtrs(self.get_K(self.Xm,self.Xn),L,upper=False)
        TrKKK = torch.sum(LslashKmn * LslashKmn)       
        reg = - 1/torch.exp(self.logsigman2) * TrKnn - TrKKK
        
        self.log_marg = log_marg
        self.reg = reg
        return(log_marg + reg)
    
    def posterior_predictive(self,test_inputs):
        
        test_inputs = torch.Tensor(test_inputs)
        Sigma = self.get_K(self.Xm,self.Xm) + 1/torch.exp(self.logsigman2) * torch.mm(self.get_K(self.Xn,self.Xm).transpose(0,1),
                                                              self.get_K(self.Xn,self.Xm))
        Sigma = Sigma+torch.eye(Sigma.shape[0])*self.jitter_factor

        #Mean
        L = torch.potrf(Sigma,upper=False)
        LslashKmnYn, _ = torch.trtrs(torch.mm(self.get_K(self.Xn,self.Xm).transpose(0,1),self.Yn),L,upper=False)
        aT, _ = torch.trtrs(self.get_K(test_inputs,self.Xm).transpose(0,1),L,upper=False)
        KxmLslash = aT.transpose(0,1)
        myq = 1/torch.exp(self.logsigman2) * torch.mm(KxmLslash,LslashKmnYn)

        #Second term of the covariance

        Kmm = self.get_K(self.Xm,self.Xm)
        Kmm = Kmm + torch.eye(Kmm.shape[0])*self.jitter_factor
        L = torch.potrf(Kmm,upper=False)
        aT, _ = torch.trtrs(self.get_K(test_inputs,self.Xm).transpose(0,1),L,upper=False)
        KxmLTslash = aT.transpose(0,1)
        LslashKmx, _ = torch.trtrs(self.get_K(test_inputs,self.Xm).transpose(0,1),L,upper=False)
        KxmKmminvKmx = torch.mm(KxmLTslash,LslashKmx)

        #Third term of the variance

        L = torch.potrf(Sigma,upper=False)
        aT, _ = torch.trtrs(self.get_K(test_inputs,self.Xm).transpose(0,1),L,upper=False)
        KxmLTslash = aT.transpose(0,1)
        LslashKmx, _ = torch.trtrs(self.get_K(test_inputs,self.Xm).transpose(0,1),L,upper=False)
        KxmSigmainvKmx = torch.mm(KxmLTslash,LslashKmx)

        #Whole covariance
        kyq = self.get_K(test_inputs,test_inputs) - KxmKmminvKmx + KxmSigmainvKmx + torch.exp(self.logsigman2)*torch.eye(test_inputs.shape[0])
        
        return(myq,kyq)
        
    
    def optimize_parameters(self,no_iters,method):
        
        # Set criterion and optimizer FOR NOW I'M GONNA USE ADAM ONLY
        '''if method == 'BFGS':
            optimizer = optim.LBFGS(self.parameters(), lr=1)  
        elif method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=0.001)
        else: 
            sys.exit('method must be either \'BFGS\' or \'Adam\'') # An exception would be better
        
        for iteration in range(no_iters):
            optimizer.zero_grad()
            loss = self.Fv() # Forward
            loss.backward() # Backward
            optimizer.step() # Optimize'''
            
        optimizer = optim.Adam(self.parameters(), lr=0.1)
        for iteration in range(no_iters):
            print(iteration)
            optimizer.zero_grad()
            loss = - self.Fv() # Forward. WHY DON'T I HAVE TO NEGATE THIS?
            loss.backward() # Backward
            optimizer.step() # Optimize
            
            if iteration%50 == 0:
                print(iteration,self.Fv())

def plot_GP(pred_mean,pred_covar,train_inputs,train_outputs,test_inputs):
    fig, ax = plt.subplots()
    plt.plot(train_inputs, train_outputs, '+k')
    pred_mean = pred_mean.data.numpy()
    pred_sd = np.sqrt(pred_covar.data.numpy().diagonal())
    plt.plot(test_inputs, pred_mean, '+b')
    plt.plot(test_inputs, np.squeeze(pred_mean) + 2*pred_sd, '+r')
    plt.plot(test_inputs, np.squeeze(pred_mean) - 2*pred_sd, '+r')
    plt.fill_between(np.squeeze(test_inputs), np.squeeze(pred_mean) + 2*pred_sd, 
                    np.squeeze(pred_mean) - 2*pred_sd, color='b', alpha=0.3)
    plt.show()

if __name__ == "__main__": 
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # load the 1D dataset
    with open('../data/1D/1D_200.pkl', 'rb') as f:
        train_inputs, train_outputs, test_inputs = pickle.load(f)
    no_train = train_outputs.size
    no_test = test_inputs.shape[0]

    # convert to torch tensors
    train_inputs = torch.Tensor(train_inputs)
    train_inputs = torch.unsqueeze(train_inputs, 1) # 1 dimensional data 
    train_outputs = torch.Tensor(train_outputs)
    test_inputs = torch.Tensor(test_inputs)
    test_inputs = torch.unsqueeze(test_inputs, 1) # 1 dimensional data

    myGP = variational_GP(train_inputs.data.numpy(), np.expand_dims(train_outputs.data.numpy(),1))
    pred_mean, pred_covar = myGP.posterior_predictive(test_inputs.data.numpy())

    # plot and save
    fig, ax = plt.subplots()
    plt.plot(torch.squeeze(train_inputs).data.numpy(), train_outputs.data.numpy(), '+k')
    pred_mean = (torch.squeeze(pred_mean)).data.numpy()
    pred_var = (torch.diag(pred_covar)).data.numpy()
    pred_sd = np.sqrt(pred_var)
    plt.plot(torch.squeeze(test_inputs).data.numpy(), pred_mean, color='b')
    plt.fill_between(torch.squeeze(test_inputs).data.numpy(), pred_mean + 2*pred_sd, 
            pred_mean - 2*pred_sd, color='b', alpha=0.3)
    filepath = 'var_GP_200.pdf'
    fig.savefig(filepath)
    plt.close()

    myGP.optimize_parameters(500,'Adam')
    pred_mean, pred_covar = myGP.posterior_predictive(test_inputs.data.numpy())

    # plot after
    fig, ax = plt.subplots()
    plt.plot(torch.squeeze(train_inputs).data.numpy(), train_outputs.data.numpy(), '+k')
    pred_mean = (torch.squeeze(pred_mean)).data.numpy()
    pred_var = (torch.diag(pred_covar)).data.numpy()
    pred_sd = np.sqrt(pred_var)
    plt.plot(torch.squeeze(test_inputs).data.numpy(), pred_mean, color='b')
    plt.fill_between(torch.squeeze(test_inputs).data.numpy(), pred_mean + 2*pred_sd, 
            pred_mean - 2*pred_sd, color='b', alpha=0.3)
    filepath = 'var_GP_200_after.pdf'
    fig.savefig(filepath)
    plt.close()