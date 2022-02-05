import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from random import randrange
from itertools import chain

def load(data):
    """Loads data without any suffix and returns data
    """
    data = pd.read_csv(f"{data}.csv")
    return data

def process(data):
    """Extracts features and target. Reshapes the data into a numpy array
    """
    y = data[:, -1].reshape(data.shape[0], 1)
    x = data[:, :-1]
    """We assume our fitting line does not necessarily crosses the origin. For that we must add a constant term
    to our prediction model. np.vstack used to create a constant matrix."""
    x = np.vstack((np.ones((x.shape[0], )), x.T)).T
    y = np.vstack((np.ones((y.shape[0], )), y.T)).T
    return x,y 

def _indexing(A, nums):
    """
    returns sub-array
    """
    if hasattr(A, 'shape'):
        return A[nums]

    return [A[j_sub] for j_sub in nums]

def dissamble_data(*arrays, sizeof_test=0.25, shufffle=True, random_seed=1):
    """
    Makes train and test arrays
    Test size is 0.25 by convention
    Seed is used to have a deterministic process
    """
    # checks
    assert 0 < sizeof_test < 1
    assert len(arrays) > 0
    length = len(arrays[0])
    for i in arrays:
        assert len(i) == length

    n_test = int(np.ceil(length*sizeof_test))
    n_train = length - n_test

    if shufffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays))

def normalize(x, y):
    x = (x-x.min())/(x.max()-x.min())
    y = (y-y.min())/(y.max()-y.min())
    
    return x, y

def confusionmatrix(actual, predicted, normalize = False):
    """
    Generates a confusion matrix
    returns 2-D matrix
    """
    pre = sorted(set(actual))
    matrix = [[0 for j_hat in pre] for j_hat in pre]
    imap   = {key: i for i, key in enumerate(pre)}
    # Generate the confusion matrix
    for p, a in zip(predicted, actual):
        matrix[imap[p]][imap[a]] += 1
    # Normalize the matrix if necessary
    if normalize:
        sigma = sum([sum(matrix[imap[i]]) for i in pre])
        matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
    return matrix

def mean(values):
    return sum(values)/float(len(values))

def variance(data, _avg):
    return sum([x-_avg]**2 for x in data)


def model(x, y, eta, iterations):
    """Actual model for Machine Learning
    Optimization: Gradient Descent"""
    size = y.size
    grad = np.zeros((x.shape[1], 1))
    cost_list = []
    for i in range (iterations):

        y_pred = np.dot(x, grad)
        """Here dot product is equivalent to squaring the matrix y-y_pred 
        but np.square can cause overflow if uint32 is used
        Since np.square does not support uint128, dot product operation can be used
        """
        squared = np.dot(y-y_pred, np.transpose(y-y_pred))
        cost = (1/(2*size))*np.sum(squared)
        # Computes the derivative of gradient vector
        d_grad = (1/size)*np.dot(x.T, y_pred - y)
        grad = grad - eta*d_grad
        cost_list.append(cost)
        
        if(i%10==0):
            print("Error: ", cost)

    # Plot the gradient to see how its decreasing.
    rng = np.arange(0, iterations)
    plt.plot(rng, cost_list)
    plt.show()
    return grad, cost_list

def accuracy(x_test, y_test, grad):
    """Return the accuracy"""
    y_pred = np.dot(x_test, grad)
    error = (1/x_test.shape[0])*np.sum(abs(y_pred-y_test))
    accuracy = round(1-error, 4)*100
    return accuracy