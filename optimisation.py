import numpy as np

def shuffle_observations(training_set):
    '''
    This function randomly shuffles the observation in the training set
    Training set is assumed to be a np.ndarray
    It assumes that the training set stores observations in the rows, and features in the columns
    The true output must be included in the array
    '''
    np.random.shuffle(training_set)

    return training_set

def get_batch(training_set, batch_size):
    '''
    This generator yields a mini-batch array from the training set

    Example use:

    for batch in get_batch(training_set, batch_size):
        print(batch)
    '''
    training_size = training_set.shape[0]
    start_batch = 0

    while start_batch < training_size:
        end_batch = start_batch + batch_size
        if end_batch >= training_size:
            batch = training_set[ start_batch: , : ]
        else:
            batch = training_set[ start_batch:end_batch , : ]
        start_batch += batch_size
        yield batch


def mini_batch(training_set, initial_weights, function, args=(), method='vanilla', learning_rate=1e-2,
            batch_size=64, max_epochs=1000, eps=1e-7, grad_tol=1e-5, shuffle=True, verbose=True, freq=10):
    '''
    This function performs a mini-batch gradient descent, and returns the learned parameters.
    Training stops after one of the following is achieved:
        - a given number of epochs has been reached
        - a smoothed version of 'function' produces two consecuvite numbers with a difference lower than a given number 
          (i.e. function has reached a stationary point)
        - the norm of the weights update is lower than a given number (i.e. the weights are not updating in a meaningful manner)

    Returns: the set of learned parameters

    === Parameters ====
    -training_set: np.array. It represents the training_set, with observations as row and features as columns.
                    Note: if "shuffle" is set as "True" (default), training_set must include the true output in the columns
    -initial_weights: array. The initial guess of the parameters
    -function: the function to be minimized.
            It must accept parameters in this form: function(weights, batch, *args)
            It must returns the cost and the gradient vector in the form (cost, gradient)
    -args (optional): a tuple containing the additional arguments taken by function, in addition to 'weights' and 'batch'
    -method (optional): a string with the chosen optimisation method
                        Available options are: "vanilla" (default), "adam" and "momentum"
    -learning_rate (optional): the learning rate to be used to perform the weights updates. Defaults to 1e-2
    -batch_size (optional): int, the size of each batch. Defaults to 64
    -max_epochs (optional): int, the maximum number of epochs before stopping the training. Defaults to 1000
    -eps (optional): the threshold level for the difference in two consecutive function values before halting the training.
                    A smoothing function is applied to the function outputs to reduce noise when using a small batch size
                    Defaults to 10^-7
    -grad_tol (optional): the threshold level for the norm of the gradient. Training halts when the norm becomes below grad_tol
                        Defaults to 10^-5
    -shuffle (optional): bool. If True, shuffles the dataset at each epoch so that each bach is different than the epoch before
                        Defaults to True
     -verbose (optional): bool. If True, prints a few information during and after training
                        Defaults to True
    -freq (optional): int. If verbose == True, prints iteration and epoch count every time epoch is a multiple of freq
                        Defaults to 10
    '''
    
    optimization_results = {
        1: 'Cost found a minimum'+' '*10,
        2: 'Weights are not updating'+' '*10,
        3: 'Max number of epochs reached'+' '*10
    }
    break_key = []
    break_flag = False

    method = method.lower()
    
    training_size = training_set.shape[0]
    if batch_size > training_size:
        batch_size = training_size

    iteration = 0
    epoch = 0
    beta_cost = 0.9
    weights = initial_weights
    update = 0
    m = 0 # adam
    v = 0 # adam


    while epoch < max_epochs:
        epoch += 1

        if shuffle:
            training_set = shuffle_observations(training_set)

        if verbose:
            if (epoch%int(freq)) == 0:
                print('Iteration: ' + str(iteration), 'Epoch: ' + str(epoch), sep=' '*5, end='\r')

        
        for batch in get_batch(training_set, batch_size):
            
            iteration += 1
            
            cost, grad = function(weights, batch, *args)
            
            
            if method == 'vanilla':
                update = -learning_rate * grad
            
            elif method == 'adam':
                beta_grad = 0.9
                beta_grad_2 = 0.999
                eps_ad = 1e-8
                
                m = beta_grad * m + (1 - beta_grad) * grad
                mt = m / (1 - beta_grad**iteration)
                
                v = beta_grad_2 * v + (1 - beta_grad_2) * (grad**2)
                vt = v / (1 - beta_grad_2**iteration)
                
                update = -learning_rate * mt / (np.sqrt(vt) + eps_ad)

            elif method == 'momentum':
                mu = 0.9
                update = mu * update - learning_rate * grad

            
            weights += update

            if iteration == 1:
                prior_avg_cost = cost * 1.5
            avg_cost = beta_cost * prior_avg_cost + (1 - beta_cost) * cost
            if np.abs(avg_cost - prior_avg_cost) < eps:
                break_flag = True
                break_key.append(1)
            prior_avg_cost = avg_cost

            
            if np.linalg.norm(grad) < grad_tol:
                break_flag = True
                break_key.append(2)

            
            if break_flag:
                break
        
        
        if epoch == max_epochs:
            break_key.append(3)
        
        if break_flag:
            break
        
            
            
    
    if verbose:
        for key in break_key:
            print(optimization_results[key])
            print('Function value: %.6f' % cost)
            print('After %d iterations in %d epochs' % (iteration, epoch) )
            print('\n')

    
    return weights

