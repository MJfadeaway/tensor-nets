import numpy as np
#import tensorflow as tf

# Author: Kejun Tang

# Date: 05/08/2018



def khatrirao(matrices, reverse=False):
    """
    khatrirao returns the Khatri-Rao product of all matrices in list "matrices".
    If reverse is true, does the product in reverse order.
    
    inputs:
    -------
           matrices    matrices list
           reverse     if reverse is true, does the operation in reverse order

    returns:
    --------
           the Khatri-Rao product matrix

    This function base on https://github.com/mrdmnd/scikit-tensor/blob/master/src/tensor_tools.py
    with a bit modified 
    """
    matorder = range(len(matrices)) if not reverse else list(reversed(range(len(matrices))))
    
    # Error checking on matrices; compute number of rows in result.
    # N = number of columns (must be same for each input)
    if matrices[0].ndim == 1:
        N = 1
    else:
        N = matrices[0].shape[1] 
    # Compute number of rows in resulting matrix
    # After the loop, M = number of rows in result.
    M = 1
    for i in matorder:
        if (matrices[i].ndim != 2) and (matrices[i].ndim != 1):
            raise ValueError("Each argument must be a matrix.")
        if (N != 1) and (N != (matrices[i].shape)[1]):
            raise ValueError("All matrices must have the same number of columns.")
        if (N == 1) and (N != matrices[i].ndim):
            raise ValueError("All matrices must have the same number of columns.")  
        M *= (matrices[i].shape)[0]
        
    # Computation
    # Preallocate result.
    P = np.zeros((M, N))
    
    # n loops over all column indices
    for n in range(N):
        # kron_vector = nth col of first matrix to consider
        if N == 1:
            kron_vector = matrices[matorder[0]][:]
        else:
            kron_vector = matrices[matorder[0]][:,n]
        # loop through matrices
        for i in matorder[1:]:
            # Compute outer product of nth columns
            #kron_vector = np.outer(matrices[i][:,n], kron_vector[:])
            if N == 1:
                kron_vector = np.kron(matrices[i][:], kron_vector[:]) 
            else:  
                kron_vector = np.kron(matrices[i][:,n], kron_vector[:])
        # Fill nth column of P with flattened result
        #P[:,n] = ab.flatten()
        P[:,n] = np.reshape(kron_vector, (M,), order='F')

    return P



def cpfactor2tensor(component_lambda, component_factor):
    """
    cpfactor2tensor returns a full tensor with its CP decomposition containing
    component_lambda and componet_factor

    inputs: 
    -------
           component_lambda       lambda in CP decomposition
           component_factor       CP factor in CP decomposition, here is a matrix list

    returns:
    --------
            a full tensor
    """
    cp_rank = len(component_lambda)
    if cp_rank == 1:
        assert component_factor[0].ndim == 1
    else:
        assert cp_rank == component_factor[0].shape[1] # consistency check

    t_siz = []
    for i, mat in enumerate(component_factor):
        t_siz.append(mat.shape[0])

    t_siz = tuple(t_siz)

    tensor_data = khatrirao(component_factor, reverse=False).dot(np.array(component_lambda))
    full_cptensor = np.reshape(tensor_data, t_siz, order='F')
    
    return full_cptensor



def tensor_unfold(T, mode):
    """
    tensor_unfold returns the mode-unfolding matrix of tensor T
    
    inputs:
    -------
           T            tensor, ndarray
          mode          mode unfolding

    returns:
    --------
           a unfolding matrix 

    """
    T_dim = len(T.shape)
    permutation = range(T_dim)
    permutation.pop(mode-1)
    permutation = [mode-1] + permutation
    m = T.shape[mode-1]
    n = np.prod(T.shape)/T.shape[mode-1]
    
    unfold_mat = np.reshape(np.transpose(T, permutation), (m,n), order = 'F')
    return unfold_mat
    


def matrix_fold(mat, mode, T_size):
    """
    matrix_fold returns the tensor with shape T_size that its mode-unfolding matrix is mat

    inputs:
    -------
          mat          unfolding matrix 
          mode         mode unfolding
         
    returns:
    --------
          a fold tensor corresponding to unfolding matrix

    """
    if type(T_size) == tuple:
        T_size = list(T_size)
    
    m = T_size.pop(mode-1)
    T_newsize = [m] + T_size
    T = np.reshape(mat, T_newsize, order='F')
    permutation = range(len(T_newsize))
    T = np.transpose(T, permutation[1:mode]+[permutation[0]]+permutation[mode::])
    
    return T



def ten_mult_mat(T, mat, mode, squeeze=False):
    """
    ten_mult_mat returns the result that tensor T multiply matrix mat: n-mode product

    inputs:
    -------
           T        tensor with size I_1*I_2*...*I_mode*...*I_N
          mat       matrix with size J * I_mode, or a vector with size I_mode * 1
          mode      mode 
        squeeze     squeeze the single dimension

    returns:
    --------
          a new tensor with size I_1*I_2*...*J*...*I_N
    """
    if T.ndim == 2: # tensor reduce to matrix
	if mode == 1:
            assert mat.shape[1] == T.shape[0]
            T = np.dot(mat, T)
         
        elif mode == 2:
            assert mat.shape[1] == T.shape[1]
            T = np.dot(mat, T.T)
            T = T.T

        else:
            raise ValueError(' Tensor dimension is 2, can not apply ten_mult_mat to mode {}. ' .format(mode))
    
    if T.ndim > 2:
        t_siz = T.shape[mode-1]
    
    
        if mat.ndim == 2: # matrix case
            mat_size = mat.shape
            assert mat_size[1] == t_siz # ensure the n-mode product can be excuted
            T_size = list(T.shape)
            T_size[mode-1] = mat_size[0]

            T = matrix_fold(np.dot(mat, tensor_unfold(T, mode)), mode, T_size)        

        elif mat.ndim == 1: # vector case
            mat_size = mat.shape
            assert mat.shape[0] == t_siz
            T_size = list(T.shape)
            T_size[mode-1] = 1
	    if squeeze:
                T = np.squeeze(matrix_fold(np.dot(mat.T, tensor_unfold(T, mode)), mode, T_size)) 
            else:
                T = matrix_fold(np.dot(mat.T, tensor_unfold(T, mode)), mode, T_size)     

        else:
            raise ValueError(' Only support tensor-matrix or tensor-vector multiplication. '

                          ' Provided array of dimension {} not in [1, 2].'.format(mat.ndim))

    return T



def multi_ten_mult_mat(T, mat_list, transpose=False, squeeze=True):
    """
    multi_ten_mult_mat returns the result that a tensor T multiply some matrices or vector stored in mat_list

    inputs:
    -------
           T         a tensor
         mat_list    matrix list
         transpose   indicate every matrix in matrix list transpose or not
         squeeze     squeeze the single dimension

    returns:
    --------
           a new tensor with proper size
            
    """
 
    for i, mat in enumerate(mat_list):
        mode = i + 1
        if transpose:
	    T = ten_mult_mat(T, mat.T, mode)
        else:         
            T = ten_mult_mat(T, mat, mode)
         
    if not squeeze:
        res = T
    else:
        res = np.squeeze(T)

    return res
        
    
    
    
    
    
    
    
