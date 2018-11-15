import numpy as np
from hopm import hopm
from tensor_operations import cpfactor2tensor


factor_mat = np.random.randn(10)
cp_factor1 = factor_mat
cp_factor2 = factor_mat
cp_factor3 = factor_mat
cp_factor4 = factor_mat
cp_factor5 = factor_mat

component_factor = [cp_factor1, cp_factor2, cp_factor3, cp_factor4]
component_lambda = [5]
cp_tensor = cpfactor2tensor(component_lambda, component_factor)

lam, rankone_factor = hopm(cp_tensor)
cp_tensor_recover = cpfactor2tensor(lam, rankone_factor)
difference_hopm = np.linalg.norm(cp_tensor_recover-cp_tensor)

print('-----test hopm-----')
print('original tensor shape', cp_tensor.shape)
print('recovered tensor shape', cp_tensor_recover.shape)
print('difference between recover and original: ', difference_hopm )



