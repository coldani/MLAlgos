import numpy as np
from scipy.optimize import minimize

class logistic_regression:
    '''
    This class represents a logistic regression classifier
    X and y are the input matrix and output vector respectively
    X: a np.ndarray matrix of observations. Each row is an observations, each column is a variable
    y: a np.array. Each element is the true output of an observation

    Methods
    train: returns the fitted parameters
    predict: returns the predicted output
    score: returns the accuracy, precision, recall and f1 scores
    '''
    
    def __init__(self, X, y = None, params = None, y_hat=None):
        self.X = X
        if y is not None:
            self.y = y.reshape(len(y),1)
        self.params = params
        self.y_hat = y_hat
        if y is not None:
            assert len(y) == X.shape[0], 'x and y have different sizes!'
    
    def sigmoid(self, X=None, params=None):
        '''
        X: np ndarray, defaults to instance X
        params: tuple, list, array, defaults to instance params
        returns: sigmoid function of matrix product of data and params
        '''
        if X is None:
            X = self.X
        if params is None:
            params = self.params

        num_params = np.size(params)
        params = np.array(params).reshape((num_params,1))
        z = np.dot(X, params)
        sigmoid = 1 / (1 + np.exp(-1*z))
        # the two lines below have been added to avoid sigmoid resulting in 0 or 1
        epsilon = 10**(-15)
        sigmoid = sigmoid - epsilon*(sigmoid==1) + epsilon*(sigmoid==0)
        return sigmoid

    def cost(self, params, X=None, y=None, regularized = True, lambda_ = 0.1, has_intercept = True):
        '''
        params: np.array of parameters
        X: np.ndarray of independent variables. shape = (# observations, # variables)
           If it includes the intercept, this must be the first column.
           Defaults to instance X
        y: np.array of dependent variables, either 1 or 0. Must be 2 dimensional: shape = (len(y),1)
           Defaults to instance y
        regularized: Boolean. If True applies regularization to the training of the model
        lambda_: regularization parameter, used only if regularized=True
        has_intercept: Boolean. Used only for regularization purposed (doesn't penalize intercept coefficient)

        returns (cost, gradient), where cost is a float and gradient is a np.array
        '''
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y
       
        m = np.size(params)
        cost = 0
        grad = np.zeros(m)
        y = y.reshape((len(y),1))

        h_x = self.sigmoid(X, params)
        cost = np.sum( -y*np.log(h_x) - (1 - y)*np.log(1 - h_x))/m
        if regularized:
            if has_intercept:
                cost += np.sum(params[1:]**2) * (lambda_ / (2*m))
            else:
                cost += np.sum(params**2) * (lambda_ / (2*m))
        
        grad = np.sum((h_x - y)*X, axis=0) / m
        if regularized:
            if has_intercept:
                grad[1:] += params[1:]*lambda_ / m
            else:
                grad += params*lambda_ / m


        return (cost, grad)

    def train(self, X=None, y=None, regularized = True, lambda_ = 0.1, has_intercept = True):
        '''
        X: the training set
        y: the correct classification (1 or 0)

        updates self.params to the trained parameters, and returns them
        '''
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        
        y = y.reshape((len(y),1))
        num_params = np.shape(X)[1]
        init_params = np.random.random(num_params)

        trained_params = minimize(self.cost, x0=init_params, args=(X, y, regularized, lambda_, has_intercept),
                                jac=True, options={'maxiter':2000, 'disp':True})
        self.params = trained_params.x

        return trained_params.x

    def predict(self, X=None, params=None):
        '''
        X: np.ndarray of independent variables. shape = (# observations, # variables)
           If it includes the intercept, this must be the first column.
           Defaults to instance X
        params: tuple, list, array, defaults to instance params

        returns the predictions, a np.ndarray of shape (len(y),1) of either 0 or 1 and updates the self.y_hat variable
        '''
        if X is None:
            X = self.X
        if params is None:
            params = self.params
        
        h_x = self.sigmoid(X, params)
        y_hat = (h_x >= 0.5)*1
        y_hat = y_hat.reshape((len(y_hat),1))
        self.y_hat = y_hat

        return y_hat

    def score(self, y=None, y_hat=None):
        '''
        returns a tuple with (precision, recall, accuracy)
        '''
        if y is None:
            y = self.y
        if y_hat is None:
            y_hat = self.y_hat

        true_positives = ((y_hat == 1) & (y == 1)).sum()
        true_negatives = ((y_hat == 0) & (y == 0)).sum()
        false_positives = ((y_hat == 1) & (y == 0)).sum()
        false_negatives = ((y_hat == 0) & (y == 1)).sum()
        total = len(y)

        accuracy = (true_positives + true_negatives) / total
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f_1 = 2 * (precision * recall) / (precision + recall)

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f_1 = f_1

        return (accuracy, precision, recall, f_1)

