import numpy as np

'''
feature_scaling: used to scale a set of features. Supports various methods - see function docstrings

split_dataset: splits a dataset into a training set, scorr-validation set and (optionally) a test set

add_intercept: quickly adds the intercept (a constant, with value equal to 1) to a dataset
'''


def feature_scaling(x, method='stand'):
    '''
    x: np ndarray, with shape (num_obs, num_var)
    method: string. Valid methods are 'stand', 'norm', 'scale'
            'stand': standardization of the variables, with 0 mean and uniot variance
            'norm': normalization of the variables, with values between [-1,1]
            'scale': normalization of the variables, with 0.5 mean and values between [0,1]
    returns: the scaled version of the array according to the chosen method
    '''
    valid_methods = ('stand', 'norm', 'scale')
    assert method in valid_methods, 'invalid scaling method'

    means = np.mean(x, axis=0)
    maxs = np.max(x, axis=0)
    mins = np.min(x, axis=0)
    ranges = maxs - mins
    std = np.std(x, axis=0)
    
    if method == 'stand':
        scaled = (x - means)/std
    elif method == 'norm':
        scaled = 2*( (x - mins)/ranges ) - 1
    elif method == 'scale':
        scaled = (x - mins)/ranges

    return scaled

def split_dataset(x, y, cross_validation_size=0.3, test_size=0.0, add_test=False, shuffle=True):
    '''
    x: a np.ndarray, with shape (num_obs, num_var)
    y: a np.array
    cross_validation_size: the size of the cross-validation set (fraction of all data)
    test_size: the size of the test set (fraction of all data). Used only if add_test = True
    add_test: boolean. If true, splits the dataset into 3 parts (training, cross-validation and test sets)
    shuffle: boolean, if True randomly shuffles the data

    returns: a training set, a cross-validation set and, if add_test = True, a test set
             y is the first column of the returning arrays

    '''
    assert len(y) == x.shape[0], 'x and y have different sizes!'
    if not(add_test):
        test_size = 0.0
    y = y.reshape((len(y),1))
    m = len(y)

    data = np.append(y,x,axis=1)
    if shuffle:
        np.random.shuffle(data)

    training_size = 1 - cross_validation_size - test_size
    training_index = int(m*training_size)
    cross_validation_index = training_index + int(m*cross_validation_size)
    
    if add_test:
        training_set = data[:training_index,:]
        cross_validation_set = data[training_index:cross_validation_index,:]
        test_set = data[cross_validation_index:,:]
        return (training_set, cross_validation_set, test_set)

    else:
        training_set = data[:training_index,:]
        cross_validation_set = data[training_index:,:]
        return (training_set, cross_validation_set)

def add_intercept(x):
    '''
    x: a np.ndarray, with observations on the rows and variables on the columns. Can be 3-dim array (e.g. RGB images)
    returns: the x array with the intercept (a constant value equal to 1) added on the first column 
    '''
    if x.ndim == 3:
        intercept = np.ones((x.shape[0], 1, x.shape[2]))
    else:
        intercept = np.ones((x.shape[0], 1))
           
    x = np.append(intercept, x, axis=1)

    return x
