import numpy as np


class AdalineGD:
    """
        ADAptive Linear Neuron classfier
        
        Parameters
        -----------
        eta: float
             Learning rate (between 0 and 1)
             
        n_iter : int
            Passes over the training dataset
            
        random_state : int
            Random number generator seed for random weight initialization
            
        Attributes
        -----------
        w_ : 1d-array
            Weights after fitting
            
        b_ : Scalar
            Bias unit after fitting
            
        losses_ : list
            Mean squarred error loss function values in each epoch
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    
    def fit(self, X, y):
        """Fitting Training Data

        Args:
            X : (array-like), shape = [n_examples, n_features]
                Training vectors, where n_examples,
                is the number of examples or observations and
                n_features is tje number of features
            y : (array-like) shape = [n_examples]
                Training Values
                
        Returns
        -----------
        self : object
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1]
                              )
        
        self.b_ = np.float_(0.0)
        self.losses_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y[:, 0] - output)
            # print(errors)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)  
        
        return self
    
    
    def net_input(self, X):
        """
            Calculate net input
        """
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """
            Compute linear function
        """
        # Activation function in this code is simply an identity function
        return X
    
    
    def predict(self, X):
        """
            Return Class Label after unit step
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    

