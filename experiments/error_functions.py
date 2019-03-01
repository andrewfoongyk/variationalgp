import torch
import numpy as np

def SMSE(pred_mean_tensor, y_tensor):
    return(torch.mean((pred_mean_tensor - y_tensor)**2)/torch.var(y_tensor))

def SNLP(pred_mean_tensor, pred_var_tensor, y_tensor, y_train_mean, y_train_sd):
    NLL = 0.5*torch.sum(torch.log(2*np.pi*pred_var_tensor)) + torch.sum((pred_mean_tensor - y_tensor)**2/(2*pred_var_tensor)) 
    # get trivial model negative LL
    no_test = y_tensor.shape[0]
    NLL_trivial = (no_test/2)*torch.log(2*np.pi*y_train_sd**2) + torch.sum((y_tensor - y_train_mean)**2/(2*y_train_sd**2))
    SLL = NLL - NLL_trivial
    MSLL = SLL/no_test
    return(MSLL)