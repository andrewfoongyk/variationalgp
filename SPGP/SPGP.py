import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SPGP(nn.Module):
    def __init__(self, input_dim, num_pseudoin, train_inputs, freeze_hyperparams=False,
                 logsigmaf2=None, logl2=None, logsigman2=None, kernel='SE'):
        super(SPGP, self).__init__()
        self.kernel = kernel

        # initialise hyperparameters
        self.input_dim = train_inputs.shape[1]  # input dimension
        self.num_pseudoin = num_pseudoin
        if freeze_hyperparams:
            self.logsigmaf2 = logsigmaf2
            self.logl2 = logl2
            self.logsigman2 = logsigman2
        else:
            self.logsigmaf2 = nn.Parameter(torch.zeros(1, dtype=torch.double))  # function variance
            self.logl2 = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.double))  # horizontal length scales
            self.logsigman2 = nn.Parameter(torch.zeros(1, dtype=torch.double))  # noise variance
        # random initialization for pseudo-inputs to subset of true datapoints
        # self.pseudoin = nn.Parameter(train_inputs.view(-1, input_dim)[torch.randperm(train_inputs.size()[0])[
        #                                                              :num_pseudoin], :input_dim])

        # initialization for toy sets
        upper = 3
        lower = 2
        input_locations = (lower - upper) * torch.rand(num_pseudoin) + upper
        self.pseudoin = nn.Parameter(torch.unsqueeze(input_locations, 1).type(torch.DoubleTensor))

    def get_K(self, inputs_col, inputs_row, length_factors):
        if self.kernel == 'SE':
            # form the kernel matrix with dimensions (input1 x input2) using squared exponential ARD kernel
            squared_distances = (inputs_col - inputs_row) ** 2            
            K = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))
        elif self.kernel == 'matern':
            abs_distances = torch.abs(inputs_col - inputs_row)
            scaled_distances = abs_distances * length_factors
            K = torch.exp(self.logsigmaf2) * torch.exp(
                -(np.sqrt(3)*torch.ones(1, device=device, dtype=torch.double)) * torch.sum(scaled_distances, 0) + torch.sum(
                    torch.log(1 + (np.sqrt(3)*torch.ones(1, device=device, dtype=torch.double)) * scaled_distances), 0))
        else:
            raise Exception('Invalid kernel name')
        return K

    def get_LL(self, train_inputs, train_outputs):
        # form the necessary kernel matrices
        Knn_diag = torch.exp(self.logsigmaf2)
        train_inputs_col = torch.unsqueeze(train_inputs.transpose(0, 1), 2)
        length_factors = (1. / (2. * torch.exp(self.logl2))).reshape(self.input_dim, 1, 1)
        pseudoin_row = torch.unsqueeze(self.pseudoin.transpose(0, 1), 1)
        pseudoin_col = torch.unsqueeze(self.pseudoin.transpose(0, 1), 2)
        Knm = self.get_K(train_inputs_col, pseudoin_row, length_factors)
        Kmn = Knm.transpose(0, 1)
        Kmm = self.get_K(pseudoin_col, pseudoin_row, length_factors)
        mKmm = torch.max(Kmm)

        L_Kmm = torch.potrf(Kmm + 1e-12*mKmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False)
        L_slash_Kmn = torch.trtrs(Kmn, L_Kmm, upper=False)[0]
        Lambda_diag = Knn_diag - (L_slash_Kmn**2).sum(0, keepdim=True).transpose(0, 1)
        diag_values = Lambda_diag + torch.exp(self.logsigman2)

        Qmm = Kmm + Kmn.matmul(Knm/diag_values)
        mQmm = torch.max(Qmm)
        L_Qmm = torch.potrf(Qmm + 1e-12*mQmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False)
        L_slash_y = torch.trtrs(Kmn.matmul(train_outputs.view(-1, 1)/diag_values), L_Qmm, upper=False)[0]

        fit = ((train_outputs.view(-1, 1))**2/diag_values).sum()-(L_slash_y**2).sum()
        log_det = 2.*torch.sum(torch.log(torch.diag(L_Qmm))) -\
            2.*torch.sum(torch.log(torch.diag(L_Kmm))) +\
            torch.sum(torch.log(diag_values))

        # get log marginal likelihood
        LL = -0.5*train_outputs.shape[0]*torch.log(2.*np.pi*torch.ones(1, device=device, dtype=torch.double)) - 0.5*log_det - 0.5*fit

        return LL

    def joint_posterior_predictive(self, train_inputs, train_outputs, test_inputs, noise=False):
        # form the necessary kernel matrices
        Knn_diag = torch.exp(self.logsigmaf2)
        train_inputs_col = torch.unsqueeze(train_inputs.transpose(0, 1), 2)
        length_factors = (1. / (2. * torch.exp(self.logl2))).reshape(self.input_dim, 1, 1)
        pseudoin_row = torch.unsqueeze(self.pseudoin.transpose(0, 1), 1)
        pseudoin_col = torch.unsqueeze(self.pseudoin.transpose(0, 1), 2)
        Knm = self.get_K(train_inputs_col, pseudoin_row, length_factors)
        Kmn = Knm.transpose(0, 1)
        Kmm = self.get_K(pseudoin_col, pseudoin_row, length_factors)
        mKmm = torch.max(Kmm)

        L_Kmm = torch.potrf(Kmm + 1e-12*mKmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False)
        L_slash_Kmn = torch.trtrs(Kmn, L_Kmm, upper=False)[0]
        Lambda_diag = Knn_diag - (L_slash_Kmn**2).sum(0, keepdim=True).transpose(0, 1)
        diag_values = Lambda_diag + torch.exp(self.logsigman2)

        Qmm = Kmm + Kmn.matmul(Knm/diag_values)
        mQmm = torch.max(Qmm)
        L_Qmm = torch.potrf(Qmm + 1e-12*mQmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False)
        L_slash_y = torch.trtrs(Kmn.matmul(train_outputs.view(-1, 1)/diag_values), L_Qmm, upper=False)[0]

        no_test = test_inputs.size()[0]

        # get cross covariance between test and train points, Ktn
        test_inputs_col = torch.unsqueeze(test_inputs.transpose(0, 1), 2)
        test_inputs_row = torch.unsqueeze(test_inputs.transpose(0, 1), 1)
        Ktm = self.get_K(test_inputs_col, pseudoin_row, length_factors)
        Kmt = Ktm.transpose(0, 1)

        # get predictive mean
        LQslashKnt = torch.trtrs(Kmt, L_Qmm, upper=False)[0]
        LKslashKnt = torch.trtrs(Kmt, L_Kmm, upper=False)[0]
        pred_mean = LQslashKnt.transpose(0, 1) @ L_slash_y

        # get predictive covariance
        Ktt = self.get_K(test_inputs_col, test_inputs_row, length_factors)
        if noise:  # add observation noise
            pred_cov = Ktt + torch.exp(self.logsigman2) * torch.eye(no_test, device=device, dtype=torch.double) +\
                       LQslashKnt.transpose(0, 1) @ LQslashKnt -\
                       LKslashKnt.transpose(0, 1) @ LKslashKnt
        else:
            pred_cov = Ktt + LQslashKnt.transpose(0, 1) @ LQslashKnt -\
                       LslashKnt.transpose(0, 1) @ LslashKnt + 1e-6 * torch.eye(no_test, device=device, dtype=torch.double)

        return pred_mean, pred_cov
