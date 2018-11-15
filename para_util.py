from __future__ import division
import numpy as np
from scipy.integrate import quad

# Author: Kejun Tang

# Last Revised: 10/23/2018



def sigmoid(x):
    """sigmoid function"""
    return 1/(np.exp(-x)+1)


def relu(x):
    """relu function"""
    return np.maximum(x, 0)


def tanh(x):
    """tanh function"""
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def score_fun_tensor( train_y, pdf='Gaussian' ):
    """score_fun_tensor"""
    """
    input:
    ------
    train_y    response 
    pdf    probability density function of input

    returns:
    --------
    a third order tensor d * d * d
    """
    if pdf != 'Gaussian':
        raise ValueError("the probability density of input is not Gaussian, not support.")
    

def fisrt_layer_bias_fourier( A, train_x, train_y, epsilon=None, pdf='Gaussian', activation='sigmoid' ):
    """
    first_layer_bias_fourier determine the bias in the first layer of fully-connected nets by Fourier method

    inputs:
    -------
           A                 weight matrix of the first layer, pre-computed by tensor decomposition
     train_x, train_y        labeled samples (x_i,y_i) in the matrix form, i = 1, 2, ..., n, each row is a sample
        epsilon              parameter for constructing the manifold in high-dimensional space, None default O(1/sqrt(n))
    probability_density      probability density function of the input in nets, default is Gaussian

    returns:
    --------
            the estimation of bias in the first layer (fully connected nets), b1

    Reference:
              M.Janzamin, H.Sedghi, and A.Anandkumar, Beating the Perils of Non-Convexity: Guaranteed Training of 
              Neural Networks using Tensor Methods.            
    """
    # check the probability density of input
    if pdf != 'Gaussian':
        raise ValueError("the probability density of input is not Gaussian, not support.")

    # consistency check for train data, train_x and train_y
    if train_x.shape[0] !=  train_y.shape[0]:
        raise ValueError("input size is not consistent with output size.") 

    num_neurons = A.shape[1]
    n = train_x.shape[0]
    d = train_x.shape[1]
    epsilon = 1/np.sqrt(n)
    
    # draw n i.i.d. frequencies w from a cap of  sphere {||w|| = 0.5, <w, a> >= (1-epsilon**2/2)/2 }
       
    # compute bias b1 in this main loop
    b1 = np.zeros(num_neurons) # initialization for b1
    for k in range(num_neurons):
        #omega = np.zeros((d, n))
        indicate_num = 0
        v = 0
        omega_sample = []
        while 1:
            # generate sample on sphere           
            sample_sphere = np.random.randn(d)
            norm_sample = np.linalg.norm(sample_sphere)
            if norm_sample >= 1e-4 # detect underflow
                sample_sphere = 0.5 * sample_sphere/norm_sample

                # judge if it on cap
                if abs(np.dot(sample_sphere, A[:,k])) > (1-epsilon**2/2.)/2.:
                    indicate_num = indicate_num + 1
                    omega_sample.append(sample_sphere)

            # if reach the number of samples of frequencies w, break
            if indicate_num >= n:
                omega_sample = np.array(omega_sample)
                break

        print('{} -th process of sampling on cap done...' . format(k+1)) 

        #omega[:,k] = sample_sphere
        # compute intermidiate variable for computing bias b1
        for kk in range(n):
            density_value = 1./np.sqrt(2*np.pi) * np.exp( -np.linalg.norm(train_x[kk,:])**2/2 )
            v = v + train_y[kk]/density_value * np.exp(complex(0, -np.dot(omega_sample[kk,:], train_x[kk,:]))) 

        # compute the fourier variable for bias b1
        v = v / n
        magnit_v = abs(v)
        phase_v = np.arccos((v/magnit_v).real)
        #phase_f
        if activation == 'sigmoid':
            fun_real = lambda x: np.cos(-.5*x)/(1+np.exp(-x))
            fun_imag = lambda x: np.sin(-.5*x)/(1+np.exp(-x))
            spectrum_real = quad(fun_real, -20, 20)
            spectrum_imag = quad(fun_imag, -20, 20)
            phase_f = np.arctan(spectrum_imag/spectrum_real)
        elif activation == 'relu':
            fun_real = lambda x: np.cos(-.5*x)*np.maximum(0, x)
            fun_imag = lambda x: np.sin(-.5*x)*np.maximum(0, x)
            spectrum_real = quad(fun_real, -20, 20)
            spectrum_imag = quad(fun_imag, -20, 20)
            phase_f = np.arctan(spectrum_imag/spectrum_real)
        elif activation == 'tanh':
            fun_real = lambda x: np.cos(-.5*x)*(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            fun_imag = lambda x: np.sin(-.5*x)*(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            spectrum_real = quad(fun_real, -20, 20)
            spectrum_imag = quad(fun_imag, -20, 20)
            phase_f = np.arctan(spectrum_imag/spectrum_real)
        else:
            raise ValueError('The activation is not supported!')

        b1[k] = 1/np.pi * (phase_v - phase_f )

    return b1


def final_layer_ridge( A, train_x, train_y, b1, reg_lam, activation='sigmoid' ):
    """
    final layer parameter computed by ridge regression

    inputs:
    -------
           A                weight matrix of the first layer, pre-computed by tensor decomposition
    train_x, train_y        labeled samples (x_i,y_i) in the matrix form, i = 1, 2, ..., n, each row is a sample
    reg_lam                 regularization parameter for ridge regression
    activation              default sigmoid active function

    returns:
    --------
    a2, a vector
    b2, a scalar
    """
    num_neurons = A.shape[1]
    num_samples = train_x.shape[0]
    # construct coeffcients matrix for ridge regression
    if activation == 'sigmoid':
        coeff_mat = (sigmoid(np.dot(A.T, train_x.T) + b1)).T
    elif activation == 'relu':
        coeff_mat = (relu(np.dot(A.T, train_x.T) + b1)).T
    elif activation == 'tanh':
        coeff_mat = (tanh(np.dot(A.T, train_x.T) + b1)).T
    else:
        raise ValueError('The activation is not supported!')

    aug_coeff_mat = np.concatenate((coeff_mat, np.ones((num_samples,1))), axis=1)
    # ridge regression
    x_sol = np.linalg.solve(aug_coeff_mat.T.dot(aug_coeff_mat) + reg_lam*np.eye(num_neurons+1), np.dot(aug_coeff_mat.T, train_y))
    a2, b2 = x_sol[:num_neurons], x_sol[num_neurons:]
    return a2, b2
