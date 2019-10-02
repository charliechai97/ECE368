import numpy as np
import matplotlib.pyplot as plt
import util
import numpy.linalg as npl

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    mu_a = 0
    Cov_a = beta*np.eye(2)
    a0 = np.arange(-1, 1.1, 0.1)
    a1 = np.arange(-1, 1.1, 0.1)
    A1, A0 = np.meshgrid(a0, a1)
    Pa = np.zeros(A1.shape)
    for i in range(0,A1.shape[0]):
        for j in range(0,A1.shape[1]):
            A = [A0[i,j],A1[i,j]]
            A = np.array(A)
            A.shape = (2,1)
            Pa[i,j] = 1/np.sqrt(2*np.pi*npl.det(Cov_a))*np.exp(-1/2*np.matmul(np.matmul((A-mu_a).T,npl.inv(Cov_a)),(A-mu_a)))
        
    plt.contour(A0,A1,Pa)
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    X = np.append(np.ones((len(x),1)),x, axis = 1)
    mu = npl.solve((np.matmul(X.T,X) + sigma2/(beta^2)),np.matmul(X.T,z))
    Cov = npl.inv(np.matmul(X.T,X) + sigma2/(beta^2))*sigma2
   
   
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    x = np.array(x)
    x.shape = (len(x),1)
    X_in = np.append(np.ones((len(x),1)),x,axis = 1)
    [mu,Cov] = posteriorDistribution(x_train,z_train,beta,sigma2)
    pred = np.matmul(X_in,mu)
    var = np.zeros(len(x))
    for i in range(0,len(x)):
        x_i = X_in[i,:] 
        var[i] = np.matmul(np.matmul(x_i.T,Cov),x_i)
        
    plt.errorbar(x,pred,yerr=var)
    plt.scatter(x_train,z_train)
    return 
    

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    NS = [1, 5, 100]
    priorDistribution(beta)
    a0 = np.arange(-1, 1.1, 0.1)
    a1 = np.arange(-1, 1.1, 0.1)
    A1, A0 = np.meshgrid(a0, a1)
    # number of training samples used to compute posterior
    for ns in NS:
        
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]
        # prior distribution p(a)
#        priorDistribution(beta)
        plt.figure()
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        Post = np.zeros(A1.shape)
        for i in range(0,A1.shape[0]):
            for j in range(0,A1.shape[1]):
                A = [A0[i,j],A1[i,j]]
                A = np.array(A)
                A.shape = (2,1)
                Post[i,j] = 1/np.sqrt(2*np.pi*npl.det(Cov))*np.exp(-1/2*np.matmul(np.matmul((A-mu).T,npl.inv(Cov)),(A-mu)))
        
        plt.contour(A0,A1,Post)
        
        plt.figure()
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
        

   

    
    
    

    
