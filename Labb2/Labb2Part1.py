import os.path
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import util


[X,z] = util.get_data_in_file('training.txt')
X = np.append(np.ones((len(X),1)),X, axis = 1)
sigma2 = 0.1
beta = 1
mu_hat = npl.solve((np.matmul(X.T,X) + sigma2/(beta^2)),np.matmul(X.T,z))
Cov_mat = npl.inv(np.matmul(X.T,X) + sigma2/(beta^2))*sigma2

mu_a = 0
Cov_a = beta*np.eye(2)

a0 = np.arange(-1, 1.1, 0.1)
a1 = np.arange(-1, 1.1, 0.1)
A1, A0 = np.meshgrid(a0, a1)

X_1 = X[0,:]
X_1.shape = (1,2)
z_1 = z[0,0]
mu_1= npl.solve((np.matmul(X_1.T,X_1) + sigma2/(beta^2)),X_1.T*z_1)
Cov_1=npl.inv(np.matmul(X_1.T,X_1) + sigma2/(beta^2))*sigma2



X_5 = X[0:5,:]
z_5 = z[0:5,0]
z_5.shape = (5,1)
mu_5=npl.solve((np.matmul(X_5.T,X_5) + sigma2/(beta^2)),np.matmul(X_5.T,z_5))
Cov_5 =npl.inv(np.matmul(X_5.T,X_5) + sigma2/(beta^2))*sigma2


X_100 = X[0:100,:]
z_100 = z[0:100,0]
z_100.shape = (100,1)
mu_100 = npl.solve((np.matmul(X_100.T,X_100) + sigma2/(beta^2)),np.matmul(X_100.T,z_100))
Cov_100 = npl.inv(np.matmul(X_100.T,X_100) + sigma2/(beta^2))*sigma2

Pa = np.zeros(A1.shape)
P_1 = np.zeros(A1.shape)
P_5 = np.zeros(A1.shape)
P_100 = np.zeros(A1.shape)
for i in range(0,A1.shape[0]):
    for j in range(0,A1.shape[1]):
        A = [A0[i,j],A1[i,j]]
        A = np.array(A)
        A.shape = (2,1)
        Pa[i,j] = 1/np.sqrt(2*np.pi*npl.det(Cov_a))*np.exp(-1/2*np.matmul(np.matmul((A-mu_a).T,npl.inv(Cov_a)),(A-mu_a)))
        P_1[i,j] = 1/np.sqrt(2*np.pi*npl.det(Cov_1))*np.exp(-1/2*np.matmul(np.matmul((A-mu_1).T,npl.inv(Cov_1)),(A-mu_1)))
        P_5[i,j] = 1/np.sqrt(2*np.pi*npl.det(Cov_5))*np.exp(-1/2*np.matmul(np.matmul((A-mu_5).T,npl.inv(Cov_5)),(A-mu_5)))
        P_100[i,j] = 1/np.sqrt(2*np.pi*npl.det(Cov_100))*np.exp(-1/2*np.matmul(np.matmul((A-mu_100).T,npl.inv(Cov_100)),(A-mu_100)))

#plt.contour(A0,A1,Pa)
#plt.savefig('prior.pdf')
#plt.figure()
#plt.contour(A0,A1,P_1)
#plt.savefig('posterior1.pdf')
#plt.figure()
#plt.contour(A0,A1,P_5)
#plt.savefig('posterior5.pdf')
#plt.figure()
#plt.contour(A0,A1,P_100)
#plt.savefig('posterior100.pdf')


#PREDICTION OF NEW:
x_in = np.arange(-4,4.2,0.2)
x_in.shape = (len(x_in),1)
X_in = np.append(np.ones((len(x_in),1)),x_in,axis = 1)
pred_1 = np.matmul(X_in,mu_1)
pred_5 = np.matmul(X_in,mu_5)
pred_100 = np.matmul(X_in,mu_100)
err_1 = np.zeros(len(x_in))
err_5 = np.zeros(len(x_in))
err_100 = np.zeros(len(x_in))
for i in range(0,len(x_in)):
    x_i = X_in[i,:] 
    err_1[i] = np.matmul(np.matmul(x_i.T,Cov_1),x_i)
    err_5[i] = np.matmul(np.matmul(x_i.T,Cov_5),x_i)
    err_100[i] = np.matmul(np.matmul(x_i.T,Cov_100),x_i)
#plt.figure()
#plt.errorbar(x_in,pred_1,yerr= err_1)
#plt.scatter(X_1[:,1],z_1)
#plt.savefig('predict1.pdf')
#plt.figure()
#plt.errorbar(x_in,pred_5,yerr= err_5)
#plt.scatter(X_5[:,1],z_5)
#plt.savefig('predict5.pdf')
#plt.figure()
#plt.errorbar(x_in,pred_100,yerr= err_100)
#plt.scatter(X_100[:,1], z_100)
#plt.savefig('predict100.pdf')

    
    
    
    