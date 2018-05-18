import numpy as np
from tensor_operations import *

# Author: Kejun Tang

# Date: 05/17/2018



def tucker(T, ranks=None):
    """
    tucker returns the Tucker decomposition of tensor T

    inputs:
    -------
           T      tensor 
          ranks   specify tensor ranks

    returns:
    --------
          core tensor,  Tucker matrix factors
    """
    n = T.ndim
    factors = []
    for i in range(n):
        mode = i + 1
        u, s, _ = np.linalg.svd(tensor_unfold(T, mode),  full_matrices=False)
        tol = s.max(axis=-1, keepdims=True) * max(tensor_unfold(T, mode).shape) * np.finfo(s.dtype).eps
        r = np.count_nonzero(s > tol, axis=-1) # numerical rank 
        u = u[:,0:r]
        factors.append(u)

    core_tensor = multi_ten_mult_mat(T, factors, transpose=True, squeeze=True)

    return core_tensor, factors



def orth_decomposition( T, reorthogonal=True, whitening=True, hopm=False ):
    """
    orthogonal tensor decomposition for a symmetric tensor T
    
    inputs:
    -------
            T       tensor
        whitening   random whitening reduce tensor to a p.s.d. matrix or exit with failure
       reorthogonal orthogonal projection of original tensor to deal with non-generic case  
 

    returns:
    --------
           orthogonal tensor factor    

    Reference:
             Tamara G Kolda. Symmetric orthogonal tensor decomposition is trivial. arXiv preprint arXiv:1503.01375, 2015.        
    """
    component_factor = []
    component_lambda = []
    n = T.ndim # extract tensor dimension
    length = T.shape[0] # length in each direction
    whitening_shape = tuple(list(T.shape)[2::])
    vector_len = np.prod(T.shape)/(length**2) # for reduced tensor and whitening tensor

    if reorthogonal:
        v, _ = np.linalg.qr(np.random.randn(length, length)) # generate random orthonormal matrix
        mat_list = []
        for i in range(n):
            mat_list.append(v)
  
        T = multi_ten_mult_mat(T, mat_list, transpose=False, squeeze=False)
        print('reorthogonal process done...')             
    else:
        T = T


    if whitening:
        #vector_len = np.prod(T.shape)/(length**2)
        count_whitening = 0
        max_count_whitening = 5000
        whitening_success_flag = False
        while 1:
            count_whitening = count_whitening + 1
            whitening_tensor = np.reshape(np.random.randn(vector_len), whitening_shape, order='F')
            C_mat = np.tensordot(T, whitening_tensor, axes=(range(n)[2::], range(n-2)))
            justify_symmetric_C = np.linalg.norm(C_mat-C_mat.T) < 1.0e-8
            if not justify_symmetric_C:
                print('whitening matrix is not symmetric anoymore, failure')
                break

            eig_val_C, eig_vec_C = np.linalg.eigh(C_mat)
            if min(eig_val_C) > -1.0e-10:                
                whitening_success_flag = True
                eig_vec_C = eig_vec_C[:, eig_val_C>1.0e-10]
                eig_val_C = eig_val_C[eig_val_C>1.0e-10]
                print('whitening process done...')
                break
            
            if count_whitening > max_count_whitening:
                print('maximal whitening iteration {} reached but whitening process failure...' .format(max_count_whitening))
                break
            #if C_mat is positive:
            #   break

    if whitening_success_flag:
        mat_list = []
        whitening_mat = np.diag(1.0/np.sqrt(eig_val_C)).dot(eig_vec_C.T)
        #print('-----whitening_mat-----', whitening_mat)
        for i in range(n):
            mat_list.append(whitening_mat)           
            
        T = multi_ten_mult_mat(T, mat_list, transpose=False, squeeze=False) 
        p = T.ndim
        new_length = T.shape[0]   
        reduced_shape = tuple(list(T.shape)[2::])
        rvector_len = np.prod(T.shape)/(new_length**2) # for reduced tensor and whitening tensor    
        reduced_tensor = np.reshape(np.random.rand(rvector_len), reduced_shape, order='F')
        B_mat = np.tensordot(T, reduced_tensor, axes=(range(p)[2::], range(p-2))) 
        justify_symmetric_B = np.linalg.norm(B_mat-B_mat.T) < 1.0e-8
        #print('-----justify_symmetric_B-----', justify_symmetric_B) 
        eig_val_B, eig_vec_B = np.linalg.eigh(B_mat)
        num_term = len(eig_val_B[abs(eig_val_B)>1.e-10])
        x_mat = np.zeros((length, num_term))
        for j in range(num_term):
            if reorthogonal:
                x = v.T.dot(eig_vec_C).dot(np.diag(np.sqrt(eig_val_C))).dot(eig_vec_B[:,j])
            else:
                x = eig_vec_C.dot(np.diag(np.sqrt(eig_val_C))).dot(eig_vec_B[:,j])
            eig_vec_B_list = []
            for jj in range(p):
                eig_vec_B_list.append(eig_vec_B[:,j])
      
            lam = multi_ten_mult_mat(T, eig_vec_B_list, transpose=False, squeeze=True)  
            component_lambda.append(lam)
            x_mat[:, j] = x

        for k in range(p):
            component_factor.append(x_mat)
 
    return component_factor, component_lambda
            
          
    

    
