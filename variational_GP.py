class variational_GP(nn.Module):
    
    def __init__(self, Xn, Yn):   # the GP takes the training data as arguments, in the form of numpy arrays, with the correct dimensions
        
        super().__init__()
        # initialise hyperparameters
        self.Xm = nn.Parameter(torch.Tensor(Xn[random.sample(range(Xn.shape[0]),30)]).type(torch.FloatTensor))
        self.Xn = torch.tensor(Xn).type(torch.FloatTensor)
        self.Yn = torch.tensor(Yn).type(torch.FloatTensor)
        self.no_inputs = Xn.shape[1]
        self.logsigmaf2 = nn.Parameter(torch.Tensor([0])) # function variance
        self.logl2 = nn.Parameter(torch.zeros(self.no_inputs)) # horizontal length scales
        self.logsigman2 = nn.Parameter(torch.Tensor([0])) # noise variance
        
    def get_K(self,inputs1,inputs2):
        
        inputs1_col = torch.unsqueeze(inputs1.transpose(0,1), 2)
        inputs2_row = torch.unsqueeze(inputs2.transpose(0,1), 1)
        squared_distances = (inputs1_col - inputs2_row)**2        
        length_factors = (1/(2*torch.exp(self.logl2))).reshape(self.no_inputs,1,1)
        K = torch.exp(self.logsigmaf2) * torch.exp(-torch.sum(length_factors * squared_distances, 0))
        return(K)
    
    def Fv_inv(self): # All the necessary arguments are instance variables, so no need to pass them

        # Compute first term (log marginal likelihood)
        M = self.get_K(self.Xm,self.Xm) + 1/torch.exp(self.logsigman2) * torch.mm(self.get_K(self.Xm,self.Xn),self.get_K(self.Xm,self.Xn).transpose(0,1))
        #print(torch.eig(M))
        L = torch.potrf(M,upper=False)
        LslashKmnYn, _ = torch.trtrs(torch.mm(self.get_K(self.Xm,self.Xn),self.Yn),L,upper=False)
        log_marg = (- Xn.shape[1] / 2 * torch.log(torch.Tensor([2*np.pi])) # - 1*D/2 * log(2*pi)
                    - 1 / 2 * (torch.log(torch.det(self.get_K(self.Xm,self.Xm) + 1/torch.exp(self.logsigman2)*torch.mm(self.get_K(self.Xm,self.Xn),self.get_K(self.Xn,self.Xm)))) 
                                          - torch.log(torch.det(self.get_K(self.Xm,self.Xm))) + Xn.shape[0]*torch.log(torch.exp(self.logsigman2))) 
                    # -1/2 (log det(self.get_K(self.Xm,self.Xm) + 1/sigma^2*self.get_K(self.Xm,self.Xn)*self.get_K(self.Xm,self.Xn).transpose(0,1)) - log det (self.get_K(self.Xm,self.Xm)) + n*log(sigma^2))
                    - 1/2*(1/torch.exp(self.logsigman2)*torch.mm(self.Yn.transpose(0,1),self.Yn) # - 1/sigma^2 * y^T y 
						   - 1/torch.exp(self.logsigman2)**2 * torch.mm(LslashKmnYn.transpose(0,1),LslashKmnYn))) # - 1/sigma^4 * b^T b
        # Compute second term (trace, regularizer)
        TrKnn = 0
        for elem in self.Xn:
            TrKnn += self.get_K(elem.unsqueeze(0),elem.unsqueeze(0))

        L = torch.potrf(self.get_K(self.Xm,self.Xm),upper=False)
        LslashKmn, _ = torch.trtrs(self.get_K(self.Xm,self.Xn),L,upper=False)
        TrKKK = torch.sum(LslashKmn * LslashKmn)       
        reg = - 1/torch.exp(self.logsigman2) * TrKnn - TrKKK
        return(log_marg + reg)
    
    def posterior_predictive(self,test_inputs):
        
        test_inputs = torch.Tensor(test_inputs)
        Sigma = self.get_K(self.Xm,self.Xm) + 1/torch.exp(self.logsigman2) * torch.mm(self.get_K(self.Xn,self.Xm).transpose(0,1),
                                                              self.get_K(self.Xn,self.Xm))

        #Mean
        L = torch.potrf(Sigma,upper=False)
        LslashKmnYn, _ = torch.trtrs(torch.mm(self.get_K(self.Xn,self.Xm).transpose(0,1),self.Yn),L,upper=False)
        aT, _ = torch.trtrs(self.get_K(test_inputs,self.Xm).transpose(0,1),L,upper=False)
        KxmLslash = aT.transpose(0,1)
        myq = 1/torch.exp(self.logsigman2) * torch.mm(KxmLslash,LslashKmnYn)

        #Second term of the covariance

        L = torch.potrf(self.get_K(self.Xm,self.Xm),upper=False)
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
        kyq = self.get_K(test_inputs,test_inputs) - KxmKmminvKmx + KxmSigmainvKmx
        
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
            loss = self.Fv_inv() # Forward
            loss.backward() # Backward
            optimizer.step() # Optimize'''
            
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for iteration in range(no_iters):
            optimizer.zero_grad()
            loss = - self.Fv_inv() # Forward. WHY DON'T I HAVE TO NEGATE THIS?
            loss.backward() # Backward
            optimizer.step() # Optimize
            
            if iteration%50 == 0:
                print(iteration,loss)
