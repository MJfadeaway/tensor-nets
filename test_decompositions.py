import numpy as np
from tensor_operations import *
from decompositions import *


#np.random.seed(10)
core_Tr = np.random.randn(3,3,3,3)
test_factor = []
for i in range(4):
    test_factor.append(np.random.randn(10,3))

Tr = multi_ten_mult_mat(core_Tr, test_factor)
T = np.random.randn(7,7,7,7)

core_tensor, factors = tucker(T, ranks=None)
core_tensor2, factors2 = tucker(Tr, ranks=None)

T1 = multi_ten_mult_mat(core_tensor2, factors2, transpose=False, squeeze=True)
T2 = multi_ten_mult_mat(core_tensor, factors, transpose=False, squeeze=True)


# test orth_decomposition
# CP rank 5
factor_mat = np.random.randn(10,5)
cp_factor1 = factor_mat
cp_factor2 = factor_mat
cp_factor3 = factor_mat
cp_factor4 = factor_mat
#cp_factor1 = np.reshape(np.repeat(factor_mat[:,0],5), (10,5))
#cp_factor2 = np.reshape(np.repeat(factor_mat[:,1],5), (10,5))
#cp_factor3 = np.reshape(np.repeat(factor_mat[:,2],5), (10,5))
component_factor = [cp_factor1, cp_factor2, cp_factor3, cp_factor4]
component_lambda = [1,1,1,1,1]
cp_tensor = cpfactor2tensor(component_lambda, component_factor)
#print('-----test_symmetric-----', np.linalg.norm(cp_tensor-cp_tensor.T))
"""
print('-----cp_tensor-----')
print(cp_tensor, cp_tensor.shape)
print('-----cp_tensor-----')
"""
orth_factor, orth_lambda = orth_decomposition( cp_tensor, reorthogonal=True, whitening=True )
print('-----orth_lambda-----', orth_lambda)
print('-----orth_factor-----', len(orth_factor))
cp_tensor_recover = cpfactor2tensor(orth_lambda, orth_factor)
difference_cptensor = np.linalg.norm(cp_tensor-cp_tensor_recover)


# print
print('----test problem 1: full rank----')
print('----original tensor shape----', T.shape)
print('----core tensor shape----', core_tensor.shape)
print('----difference----', np.linalg.norm(T-T2))

print('----test problem 2: reduce rank----')
print('----original tensor shape----', Tr.shape)
print('----core tensor shape----', core_tensor2.shape)
print('----difference----', np.linalg.norm(Tr-T1))


print('----- test orth_decomposition-----')
print('----original tensor shape----', cp_tensor.shape)
print('----orth_tensor shape----', cp_tensor_recover.shape)
print('difference: {:.8}'. format(difference_cptensor))


