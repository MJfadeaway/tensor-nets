import numpy as np
from tensor_operations import *




def hopm( T, tol=1.0e-6 ):
    """
    high order power method for tensor T
    
    inputs:
    -------
           T           tensor
          tol          tolerance for iteration

    returns:
    --------
            rank one CP tensor factor and corresponding lambdas
    
    """
