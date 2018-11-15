import numpy as np
from tensor_operations import *

# Author: Kejun Tang

# Date: 05/19/2018



def hopm( T, tol=1.0e-6, maxiter=500 ):
    """
    high order power method for tensor T
    find a rank one approximation of tensor T
    
    inputs:
    -------
           T           tensor
          tol          tolerance for iteration

    returns:
    --------
            rank one CP tensor factor and corresponding lambda
    
    """
    tensor_order = T.ndim
    rankone_factor = [] # rank one factor of tensor T, a list contains tensor_order vectors
    # initialization lambda list for justifying the convergence
    lam_list = []
    # intialization rank one factor through HOSVD
    for i in range(tensor_order):
        u, _, _ = np.linalg.svd(tensor_unfold(T, i+1))
        initial_vector = u[:,0]
        rankone_factor.append(initial_vector)
        lam_list.append(np.linalg.norm(initial_vector))
     
    converged = False
    # main loop for computing rank one CP factor
    while not converged:
        lam_previous = lam_list
        # high order power iteration process with matrix form
        # Kronecker product 
        for j in range(tensor_order):
            kron_vector = 1
            for k in range(tensor_order):
                if k == j:
                    rankone_factor[k] = 1
                # compute the Kronecker product except for j
                kron_vector = np.kron(kron_vector, rankone_factor[k])

            # update j-th rank one CP factor and lambda list
            rankone_factor[j] = tensor_unfold(T, j+1).dot(kron_vector)
            lam_list[j] = np.linalg.norm(rankone_factor[j])
            # normalize
            rankone_factor[j] = rankone_factor[j] / lam_list[j]

        # justify convergence through lambda list
        converged = all(abs(np.array(lam_list)-np.array(lam_previous)) < tol)

    # computing the coefficients lambda in rank one tensor
    lam = [multi_ten_mult_mat(T, rankone_factor, transpose=False, squeeze=True)]
    

    return lam, rankone_factor

    
