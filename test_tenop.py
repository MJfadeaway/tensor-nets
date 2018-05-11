import numpy as np
from tensor_operations import *

np.random.seed(10)

mode = 1
T = np.random.randn(7,7,7,7)
mat = np.random.randn(10,7)

# test tensor_unfold and matrix_fold

mat_unfold = tensor_unfold(T, mode)
#print(mat_unfold.shape)
T1 = matrix_fold(mat_unfold, mode, T.shape)
testerror_unfold = np.linalg.norm(T-T1)
print('---tensor_unfold,matrix_fold,error----')
print('test error fold unfold: {:.8}'. format(testerror_unfold))

# test ten_mult_mat


T = np.zeros((3,4,2))
T[:,:,0] = np.reshape(np.arange(1,13,1),(3,4),order='F')
T[:,:,1] = np.reshape(np.arange(13,25,1),(3,4),order='F')
u = np.reshape(np.arange(1,7,1),(2,3),order='F')
v = np.array([1,2,3,4])
vv = np.random.randn(6,4)
w = np.random.randn(5,2)
mat_list = [u, v, w]
u1 = np.random.randn(3)
u2 = np.random.randn(4)
u3 = np.random.randn(2)
mat_list1 = [u1, u2, u3]
Tnew = ten_mult_mat(T, u, mode)
Tv = ten_mult_mat(T, v, 2, squeeze=True)
T3 = multi_ten_mult_mat(T, mat_list)
lam = multi_ten_mult_mat(T, mat_list1)

Tv_true = np.array([[70,190],[80,200],[90,210]])
print('----squeeze false for Tv----')
print(ten_mult_mat(T, v, 2))
print(Tv_true)
print('-----difference Tv-----', np.linalg.norm(Tv-Tv_true))

Y = np.zeros((2,4,2))
Y[:,:,0] = np.array([[22,49,76,103],[28,64,100,136]])
Y[:,:,1] = np.array([[130,157,184,211],[172,208,244,280]])
testerror_tmm = np.linalg.norm(Y-Tnew)
testerror_tmv = np.linalg.norm(Tv-Tv_true)


# test cpfactor2tensor
cp_factor1 = np.reshape(np.arange(1,9,1), (4,2), order='F')
cp_factor2 = np.reshape(np.arange(1,11,1), (5,2), order='F')
cp_factor3 = np.reshape(np.arange(1,7,1), (3,2), order='F')
component_factor = [cp_factor1, cp_factor2, cp_factor3]
component_lambda = [2,3]
cp_tensor = np.zeros((4,5,3))
cp_tensor[:,:,0] = np.array([[362,424,486,548,610],[436,512,588,664,740],[510,600,690,780,870],[584,688,792,896,1000]])
cp_tensor[:,:,1] = np.array([[454,533,612,691,770],[548,646,744,842,940],[642,759,876,993,1110],[736,872,1008,1144,1280]])
cp_tensor[:,:,2] = np.array([[546,642,738,834,930],[660,780,900,1020,1140],[774,918,1062,1206,1350],[888,1056,1224,1392,1560]])
full_tensor = cpfactor2tensor(component_lambda, component_factor)
error_cp = np.linalg.norm(full_tensor-cp_tensor)


# test tensor reduced to scalar by multiply vector list
rankone_tensor = cpfactor2tensor([1], [u1, u2, u3])
lam_1 = np.tensordot(rankone_tensor, T, axes=([0,1,2], [0,1,2]))
print('-----test tensor reduced to scalar-----', lam_1-lam)

print('----Tnew shape-----')
print(Tnew.shape)
print('-----Tv shape------')
print(Tv.shape)
print('----T3 shape----')
print(T3.shape)
print('----error tmm----')
print('error for ten_mult_mat: {:.8}'. format(testerror_tmm))
print('----error tmv----')
print('error for ten_mult_vec: {:.8}'. format(testerror_tmv))

print('----lambda----')
print(lam)
print('---lambda shape---',lam.shape)

print('-----test cpfactor2tensor-----')
print('-----tensor shape-----', full_tensor.shape)
print('error for cpfactoretensor: {:.8}'. format(error_cp))
#ten_mult_mat(T, np.random.randn(2,3,4), 2) # for raising ValueError



