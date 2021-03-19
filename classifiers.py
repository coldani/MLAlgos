import numpy as np
import optimisation as opt

'''
logistic_regression: a class that implements the logistic regression classifier.
                    It allows to fit the parameters, predict results and compute a few performance metrics to evaluate the predictions
neural_network: a class that implements a Neural Network classifier.
                The NN can have any number of hidden layers and nodes, and currently supports four activation functions: relu,
                leaky relu, softmax (for output layer only) and sigmoid. 
'''

class logistic_regression:
    '''
    This class represents a logistic regression classifier

    ==== Parameters ====
    X and y are the training set and output vector respectively
    -X: np.ndarray. Each row is an observations, each column is a variable
    -y: np.array. Each element is the true output value of an observation (either 1 or 0)
    -has_intercept: Boolean.  Used only for regularization purposes (it doesn't penalize intercept coefficient)
    -regularized: Boolean. If True, applies regularization to the training of the model

    ==== Methods ====
    -train: returns the fitted parameters, and saves them as instance variable
    -predict: returns the predicted output, and saves it as instance variable
    -score: returns the accuracy, precision, recall and f1 scores
    '''
    
    def __init__(self, X, y = None, has_intercept=True, regularized=True, params = None, y_hat=None):
        
        if len(X.shape)==1:
            X = X.reshape(len(X),1)
        self.X = X
        if y is not None:
            self.y = y.reshape(len(y),1)
        self.has_intercept = has_intercept
        self.regularized = regularized
        self.params = params
        self.y_hat = y_hat
        if y is not None:
            assert len(y) == X.shape[0], 'x and y have different sizes!'
    
    def _sigmoid(self, X=None, params=None):
        '''
        X: np ndarray, defaults to instance X
        params: tuple, list, array, defaults to instance params
        returns: sigmoid function of the matrix product of data and params
        '''
        if X is None:
            X = self.X
        if params is None:
            params = self.params

        num_params = np.size(params)
        params = np.array(params).reshape((num_params,1))
        z = np.dot(X, params)
        sigmoid = 1 / (1 + np.exp(-1*z))
        
        # the two lines below have been added to avoid sigmoid resulting in exactly 0 or 1
        epsilon = 10**(-15)
        sigmoid = sigmoid - epsilon*(sigmoid==1) + epsilon*(sigmoid==0)
        
        return sigmoid

    def _cost(self, params, X=None, y=None, regularized = None, lambda_ = 0.1, has_intercept = None):
        '''
        params: np.array of parameters
        X: np.ndarray of independent variables. shape = (# observations, # variables)
           If X includes the intercept, this must be the first column.
           Defaults to instance X
        y: np.array of dependent variables, either 1 or 0. Must be 2 dimensional: shape = (len(y),1)
           Defaults to instance y
        regularized: Boolean. If True, applies regularization to the training of the model
        lambda_: regularization parameter, used only if regularized=True
        has_intercept: Boolean. Used only for regularization purposes (it doesn't penalize intercept coefficient)

        returns (cost, gradient), where cost is a float, and gradient is a np.array
        '''
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if regularized is None:
            regularized = self.regularized
        if has_intercept is None:
            has_intercept = self.has_intercept       

        m = np.size(params)
        cost = 0
        grad = np.zeros(m)
        y = y.reshape((len(y),1))

        h_x = self._sigmoid(X, params)
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

    def _wrapper(self, params, batch, regularized, lambda_, has_intercept):

        X = batch[: , :-1]
        y = batch[: , -1]

        cost, grad = self._cost(params, X, y, regularized, lambda_, has_intercept)

        return (cost, grad)


    def train(self, X=None, y=None, regularized = None, lambda_ = 0.1, has_intercept = None):
        '''
        X: the training set
        y: the correct classification (1 or 0)

        returns the trained parameters, and updates self.params
        '''
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if regularized is None:
            regularized = self.regularized
        if has_intercept is None:
            has_intercept = self.has_intercept       
        
        y = y.reshape((len(y),1))
        num_params = np.shape(X)[1]
        init_params = np.random.random(num_params)
        training_set = np.concatenate((X, y), axis=1)
        trained_params = opt.mini_batch(training_set, init_params, self._wrapper, args=(regularized, lambda_, has_intercept),
                                         batch_size = np.inf, method='adam')

        self.params = trained_params

        return trained_params

    def predict(self, X=None, params=None):
        '''
        X: np.ndarray of independent variables. shape = (# observations, # variables)
           If it includes the intercept, this must be the first column.
           Defaults to instance X
        params: tuple, list, array, defaults to instance params

        returns the predicted y variable (np.ndarray of shape (len(y),1) of either 0 or 1) and updates the self.y_hat variable
        '''
        if X is None:
            X = self.X
        if params is None:
            params = self.params
        
        h_x = self._sigmoid(X, params)
        y_hat = (h_x >= 0.5)*1
        y_hat = y_hat.reshape((len(y_hat),1))
        self.y_hat = y_hat

        return y_hat

    def score(self, y=None, y_hat=None):
        '''
        returns a tuple with (accuracy, precision, recall, f1)
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
      

class neural_network:
    '''
    This class implements a Neural Network classifier, with customizable layers. 

    ==== Parameters ====
    -X: np array. The input array. Observations in the rows, variables in the columns
    -y: np array. The output array Values must be integers, with values range depending on num_labels.
                I.e. [0,1] or [0,1,2] or generally [0, ..., num_labels-1]
    -num_labels: integer, defaults to 2. Represents the number of possible values for the output nodes (2 for [0,1], 3 for [0,1,2], etc)
    -size_hidden_layers: list of integers. Each value is the number of nodes for each hidden layer. 
                        The number of hidden layers is given by the number of elements of this list
    -add_bias: boolean, defaults to True. If True the NN will add a bias (a vector of ones) to the input and hidden layer(s)
    -regularization: boolean, defaults to True. If True the NN will add L2 regularization during training
    -lambda_: float, defaults to 0.1. Represents the regularization parameter (which is then scaled by 1/(2*m)). Used only if regularization=True
    -activation_funct: list (or tuple) of strings, representing the activation functions of each layer.
                    Valid values are: 'sigmoid', 'relu', 'leaky relu', 'softmax'
                    Activation of output layer must be one of 'sigmoid' or 'softmax'
                    Lenght of list must be (num_layers - 1)
    -weights: dictionary containing the weight matrixes in the form of np ndarray. Defaults to None
            The shape of the weights matrix from layer i to layer i+1 is (# nodes layer i+1)*(# nodes layer i), with the possible addition of the bias weights (first column)

    ==== Methods ====
    -train: returns the fitted weights, and saves them as instance attribute. By default uses a mini-batch GD method
    -predict: returns the predicted output, and saves it as instance attribute
    -score: returns the [accuracy, prediction, recall] scores, and saves them as an instance attribute

    ==== Example ====
    import neural_network as nn
    import various as vr
    import matplotlib.pyplot as plt
    from sklearn import datasets # we will use the breast cancer dataset from sklearn
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target

    # let's standardize the features to have 0 mean and unit variance
    X_scaled = vr.feature_scaling(X)

    # features split between training set and validation set
    ts, vs = vr.split_dataset(X_scaled, y)
    y_ts = ts[:,0]
    x_ts = ts[:,1:]
    y_vs = vs[:,0]
    x_vs = vs[:,1:]

    # model initialization
    nn_model = nn.neural_network(x_ts, y_ts, num_labels=2, size_hidden_layers=[30, 15], 
            add_bias=True, regularization=True, lambda_=0.1, activation_funct=('relu', 'relu', 'sigmoid'))
    # model training
    weights = nn_model.train()

    # predictions and performance measures on the training and validation sets
    tr_prediction = nn_model.predict()
    tr_score = nn_model.score()
    v_prediction = nn_model.predict(X=x_vs)
    v_score = nn_model.score(y=y_vs, prediction=v_prediction)
    
    print('')
    print('Training set score is: '+str(tr_score))
    print('Validation set score is: '+str(v_score))
    
    # let's have a look at the cost at every iteration
    plt.plot(nn_model.cost_evolution)
    '''

    def __init__(self, X, y=None, num_labels=2, size_hidden_layers=[], add_bias=True,
                regularization=True, lambda_=0.1, activation_funct=[], weights=None):
        
        #available functions
        self.dict_functions = {
            'sigmoid': self._sigmoid,
            'relu': self._ReLU,
            'leaky relu': self._leaky_ReLU,
            'softmax': self._softmax
            }

        self.dict_derivatives = {
            'sigmoid': self._sigmoid_derivative,
            'relu': self._ReLU_derivative,
            'leaky relu': self._leaky_ReLU_derivative
            }

        #define input and output layers
        self.X = X
        if y is not None:
            self.y = y.reshape((len(y), 1))
            self.y_matrix = np.zeros((len(y), num_labels))
            for obs in range(len(y)):
                self.y_matrix[obs, int(y[obs])] += 1

        #create useful instance variables for describing the NN structure
        self.num_obs = X.shape[0]
        self.num_inputs = X.shape[1] #the number of input features
        self.num_labels = num_labels
        self.num_layers = len(size_hidden_layers) + 2 # the number of NN layers, including input and output layers
        self.size_layers = np.append(self.num_inputs, size_hidden_layers) #array with the size of each NN layer (including input and output)
        self.size_layers = np.append(self.size_layers, num_labels)
        self.size_layers = list(map(int, self.size_layers))
        self.add_bias = add_bias
        self.regularization = regularization
        self.lambda_ = lambda_
        self.activation_funct = activation_funct
        if weights is not None:
            self.weights = weights
        else:
            self.weights = {}

        self.a = {}
        self.z = {}
        self.delta = {}

        #errors tests
        if y is not None:
            assert len(y) == X.shape[0], 'X and y have different sizes!'
        assert len(size_hidden_layers)+1 == len(activation_funct), 'Please check the activation functions!'
        if weights is not None:
            assert len(weights) == self.num_layers-1, 'Wrong number of weights matrices!'
        assert 'softmax' not in activation_funct[:-1], 'Softmax can only be as the activation function for the output'
        assert ((activation_funct[-1] == 'softmax') or (activation_funct[-1] == 'sigmoid')), 'Please only use \'softmax\' or \'sigmoid\' as activation function for the output layer'
        
        #for debugging
        self.cost_evolution = []
    
    def _sigmoid(self, z):
        '''
        z: np ndarray
        returns: sigmoid function of z
        '''
        _sigmoid = 1 / (1 + np.exp(-1*z))
        # the two lines below have been added to avoid sigmoid resulting in exactly 0 or 1
        epsilon = 10**(-15)
        _sigmoid = _sigmoid - epsilon*(_sigmoid==1) + epsilon*(_sigmoid==0)
        return _sigmoid

    def _sigmoid_derivative(self, z):
        derivative = self._sigmoid(z)*(1 - self._sigmoid(z))
        return derivative

    def _ReLU(self, z):
        '''
        z: np ndarray
        returns: rectified linear function of z
        '''
        relu_ = np.maximum(0.0, z)
        return relu_

    def _ReLU_derivative(self, z):
        derivative = 0.0 + 1.0*(z>0)
        return derivative

    def _leaky_ReLU(self, z):
        '''
        z: np ndarray
        returns: modified rectified linear function of z
        '''
        a = 0.01
        leaky_relu_ = np.maximum(a*z, z)
        return leaky_relu_

    def _leaky_ReLU_derivative(self, z):
        a = 0.01
        derivative = a*(z<=0) + 1.0*(z>0)
        return derivative

    def _softmax(self, z):
        '''
        z: np ndarray, with observations in the rows, labels in the columns
        returns: softmax function of z 
        '''
        z -= np.max(z, axis=1, keepdims=True) # trick for numerical stability
        softmax = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return softmax

    def _forward_propagation(self, weights=None, X=None, save_attributes=False):
        '''
        Calculates all a[layer] vectors, where a[0] is the input layer and a[num_layers] is the fitted output layer
        Saves a as a dictionary (self.a)
        '''
        if X is None:
            X = self.X
        if weights is None:
            weights = self.weights
        z = {}
        a = {}
        a[0] = X
        for layer in range(self.num_layers-1):
            if self.add_bias:
                bias = np.ones((X.shape[0],1))
                z[layer+1] = np.append(bias, a[layer],axis=1)
            else:
                z[layer+1] = a[layer]
            z[layer+1] = np.matmul(z[layer+1], weights[layer+1].T)
            funct = self.dict_functions[self.activation_funct[layer]]
            a[layer+1] = funct(z[layer+1])
        
        y_hat = a[self.num_layers-1]

        #### to avoid numerical issues
        epsilon = 1e-10
        y_hat = np.where(y_hat==0, y_hat+epsilon, y_hat)
        ####

        if save_attributes:
            self.y_hat = y_hat
            self.a = a
            self.z = z
        return y_hat
        
    def _cost(self, weights=None, X=None, y=None, save_attributes=False, one_hot=False):
        '''
        Returns the loss function of the neural network (cross entropy)
        '''
        if X is None:
            X = self.X
            num_obs = self.num_obs
        else:
            num_obs = X.shape[0]
        if weights is None:
            weights = self.weights
        if y is None:
            y_matrix = self.y_matrix
        else:
            if one_hot: # use one_hot=True when 'y' is given as a one_hot encoded matrix
                y_matrix = y
            else:
                y = y.reshape((len(y), 1))
                y_matrix = np.zeros((len(y), self.num_labels))
                for obs in range(len(y)):
                    y_matrix[obs, int(y[obs])] += 1

        y_hat = self._forward_propagation(weights=weights, X=X, save_attributes=save_attributes)
        cost = np.sum( (-1)*y_matrix*np.log(y_hat) ) / num_obs
        if self.regularization:
            l_2_reg = 0.0
            m = 0 #number of parameters
            for key in weights:
                if self.add_bias:
                    l_2_reg += np.sum(weights[key][:,1:]**2)
                else:
                    l_2_reg += np.sum(weights[key]**2)
                m += weights[key].size
            self.m=m
            cost += 1/(2*m) * self.lambda_ * l_2_reg

        self.cost_evolution.append(cost) # for debugging
        
        return cost

    def _back_propagation(self, a=None, z=None, weights=None, y=None, one_hot=False):
        '''
        a: a dict of a[layer] vectors
        '''

        if a is None:
            a = self.a
        if z is None:
            z = self.z
        if weights is None:
            weights = self.weights
        if y is None:
            y_matrix = self.y_matrix
            num_obs = self.num_obs
        else:
            num_obs = y.shape[0]
            if one_hot:
                y_matrix = y
            else:
                y = y.reshape((len(y), 1))
                y_matrix = np.zeros((len(y), self.num_labels))
                for obs in range(len(y)):
                    y_matrix[obs, int(y[obs])] += 1


        delta = {}
        grad = {}
        list_layers = list(range(self.num_layers))

        #external layer
        if self.activation_funct[-1] == 'sigmoid':
            delta[self.num_layers-1] = a[self.num_layers-1]*y_matrix - y_matrix
        elif self.activation_funct[-1] == 'softmax':
            delta[self.num_layers-1] = a[self.num_layers-1] - y_matrix
        if self.add_bias:
            bias = np.ones((num_obs,1))
            a_l_1 = np.append(bias, a[self.num_layers-2], axis=1) # this is a[penultimate_layer]
        else:
            a_l_1 = a[self.num_layers-2]
        grad[self.num_layers-1] = (1/num_obs)*np.matmul(delta[self.num_layers-1].T,a_l_1)
        if self.regularization:
            weights_temp = weights[self.num_layers-1]
            if self.add_bias:
                weights_temp[:,0] = 0.0
            grad[self.num_layers-1] += weights_temp*(self.lambda_/self.m)

        #hidden layers
        for layer in list_layers[-2:0:-1]:
            deriv = self.dict_derivatives[self.activation_funct[layer-1]]

            if self.add_bias:
                weights_l1 = weights[layer+1][:,1:]
            else:
                weights_l1 = weights[layer+1]
            delta[layer] = np.matmul(delta[layer+1],weights_l1)
            delta[layer] = delta[layer]*deriv(z[layer])
            if self.add_bias:
                bias = np.ones((num_obs,1))
                a_l_1 = np.append(bias, a[layer-1], axis=1)
            else:
                a_l_1 = a[layer-1]
            grad[layer] = (1/num_obs)*np.matmul(delta[layer].T, a_l_1)
            if self.regularization:
                weights_temp = weights[layer]
                if self.add_bias:
                    weights_temp[:,0] = 0.0
                grad[layer] += weights_temp*(self.lambda_/self.m)

        self.grad = grad
        
        return grad
    
    def _wrapper(self, weights_unrolled, batch):
        
        X = batch[: , :-self.num_labels]
        y = batch[: , -self.num_labels:]
        
        #reshape weights vector
        reshaped_weights = {}
        for layer in range(1,self.num_layers):
            size = (self.size_layers[layer-1]+self.add_bias)*self.size_layers[layer]
            reshaped_weights[layer] = weights_unrolled[:size].reshape(((self.size_layers[layer], self.size_layers[layer-1]+self.add_bias)))
            weights_unrolled = weights_unrolled[size:]
        cost = self._cost(weights=reshaped_weights, X=X, y=y, save_attributes=True, one_hot=True)
        
        gradient = self._back_propagation(weights=reshaped_weights, y=y, one_hot=True)
        gradient_unrolled = np.asarray([])
        for layer in range(1,self.num_layers):
            gradient_layer_unrolled = gradient[layer].flatten()
            gradient_unrolled = np.append(gradient_unrolled, gradient_layer_unrolled)
        
        return (cost, gradient_unrolled)

    def train(self, method='adam', learning_rate=1e-2, batch_size=64, max_epochs=1000, eps=0, grad_tol=1e-5, shuffle=True, verbose=True, freq=1):
        '''
        This function trains the neural network and returns the trained weights
        Trained weights are also saved as instance variable

        Parameters (optional):
        -method: string, accepts the following methods: 'vanilla', 'adam', 'momentum'
                Defaults to 'adam'
        -learning_rate: float, represents the value of the learning rate
                Defaults to 0.01
        -batch_size: size of the batch. Allows to implement SGD (batch_size=1), mini-batch GD (e.g. batch_size=64) or 
                full batch gradient descent (batch_size=self.X.shape[0])
                Defaults to 64
        -max_epochs: int, maximum numbers of epochs before halting the training
                Defaults to 1000
        -eps: float, a parameter to halt earlier the training when the cost function converges to a minimum - see description and 
                implementation in the optimisation function
                Defaults to 0 (i.e., this method is not used to halt the training)
        -grad_tol: float, a parameter used to halt the training earlier. Training is stopped when the norm of the gradient vector
                becomes smaller than grad_tol
                Defaults to 1e-5
        -shuffle: bool, if True the training set is randomly re-shuffled at every epoch
                Defaults to True
        -verbose: bool, if True print information during and after training
                Defaults to True
        -freq: int, used to set the frequency (at epoch level) at which training info is printed. Used only if verbose==True
                Defaults to 1 (information printed at every epoch)
        '''

        self.cost_evolution = [] #for debugging
        options = {
            'method': method,
            'learning_rate': learning_rate, 
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'eps': eps,
            'grad_tol': grad_tol,
            'shuffle': shuffle,
            'verbose': verbose,
            'freq': freq
        }


        #initialising weights
        unrolled_initial_weights = np.asarray([])
        for layer in range(1, self.num_layers):
            unrolled_layer_weights = np.random.random((self.size_layers[layer]*(self.size_layers[layer-1]+self.add_bias)))
            unrolled_layer_weights = unrolled_layer_weights*np.sqrt(2/(self.size_layers[layer]+self.size_layers[layer-1]))
            unrolled_initial_weights = np.append(unrolled_initial_weights, unrolled_layer_weights)
        
        #creating 'training_set'
        training_set = np.concatenate((self.X, self.y_matrix), axis=1)

        weights_unrolled = opt.mini_batch(training_set, unrolled_initial_weights, self._wrapper, **options)
        
        reshaped_weights = {}
        for layer in range(1,self.num_layers):
            size = (self.size_layers[layer-1]+self.add_bias)*self.size_layers[layer]
            reshaped_weights[layer] = weights_unrolled[:size].reshape(((self.size_layers[layer], self.size_layers[layer-1]+self.add_bias)))
            weights_unrolled = weights_unrolled[size:]       
        self.weights = reshaped_weights

        #the following line calculates y_hat using the entire training set and saves it as an instance variable
        y_hat = self._forward_propagation(save_attributes=True)

        return self.weights
    
    def predict(self, X=None, weights=None):
        '''
        given X and the weights, this methods predicts the output on the basis of the NN structure specified when initialising the instance
        -X: np array. The input array. Must have different observations in the rows, and variables in the columns
        -weights: dictionary with the weight matrixes, as np ndarray
        Size of weights (# of dict keys) is (num_layers - 1)
        '''
        if X is None:
            X = self.X
            num_obs = self.num_obs
        else:
            num_obs = X.shape[0]
        if weights is None:
            weights = self.weights
        y_hat = self._forward_propagation(weights=weights, X=X, save_attributes=False)
        prediction = np.argmax(y_hat, axis=1).reshape((num_obs,1))
        self.prediction = prediction
        
        return prediction
    
    def score(self, y=None, prediction=None):
        '''
        returns a dict with (accuracy, precision, recall)
        '''
        if y is None:
            y = self.y
        else:
            y = y.reshape((len(y), 1))
        if prediction is None:
            prediction = self.prediction
        else:
            prediction = prediction.reshape((len(prediction), 1))
        assert len(y) == len(prediction), 'y and prediction must have the same number of observations'

        n = len(y)
        dispersion_matrix = np.zeros((self.num_labels, self.num_labels))
        for obs in range(n):
            true_label = int(y[obs])
            predicted_label = int(prediction[obs])
            dispersion_matrix[predicted_label, true_label] += 1

        num_correct_classified = (y==prediction).sum()        
        accuracy = num_correct_classified/n
        precision = []
        recall = []
        for label in range(self.num_labels):
            true_positive = dispersion_matrix[label, label]
            predicted = dispersion_matrix[label, :].sum()
            true_obs = dispersion_matrix[:, label].sum()
            prec = true_positive / predicted
            rec = true_positive / true_obs
            precision.append(prec)
            recall.append(rec)
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall

        score = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
        }
        self.score_dict = score
        
        return score
