import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    pi_male = 0.5
    pi_female = 0.5
    x_male = x[y==1,:]
    x_female = x[y==2,:]
    mu_male = np.mean(x_male,axis = 0)
    mu_female = np.mean(x_female,axis = 0)
    cov_male = np.zeros((2,2))
    cov_female = np.zeros((2,2))
    cov = np.zeros((2,2))
    for i in range(0,len(x_male)):
        cov_male +=  np.outer(x_male[i,:]-mu_male,(x_male[i,:]-mu_male))
    for i in range(0,len(x_female)):  
#        print(np.outer((x_female[i,:]-mu_female), x_female[i,:]-mu_female))
        cov_female += np.outer(x_female[i,:]-mu_female,(x_female[i,:]-mu_female))
        
    cov = (cov_male + cov_female)/len(x)
    cov_male = cov_male/(len(x_male))
    cov_female = cov_female/len(x_female)
    
    plt.figure(1)
    
    plt.scatter(x_male[:,0],x_male[:,1],color = 'blue')
    plt.scatter(x_female[:,0],x_female[:,1], color = 'red')
    X,Y = np.meshgrid(np.arange(50,81), np.arange(80,281))
#    XY = np.append(X.flatten(),Y.flatten())
#    XY = XY.reshape(len(X.flatten()), 2)
    Z_male = np.zeros(X.shape)
    Z_female = np.zeros(X.shape)
    cov_inv = npl.inv(cov)
    lda_male = np.zeros(X.shape)
    lda_female = np.zeros(X.shape)
    
    #LDA
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[1]):
            XY = [X[i,j], Y[i,j]]
            Z_male[i,j] = 1/(2*np.pi*np.sqrt(npl.det(cov)))*np.exp(-1/2*np.matmul(np.matmul((XY-mu_male),cov_inv),(XY-mu_male).T))
            Z_female[i,j] = 1/(2*np.pi*np.sqrt(npl.det(cov)))*np.exp(-1/2*np.matmul(np.matmul((XY-mu_female),cov_inv),(XY-mu_female).T))
            
            lda_male[i,j] = np.matmul(np.matmul(XY,cov_inv),mu_male) - \
                    1/2*np.matmul(np.matmul(mu_male.T,cov_inv),mu_male) + np.log(pi_male)
                    
            lda_female[i,j] = np.matmul(np.matmul(XY,cov_inv),mu_female) - \
                    1/2*np.matmul(np.matmul(mu_female.T,cov_inv),mu_female) + np.log(pi_female)
            
    plt.contour(X,Y,Z_male)
    plt.contour(X,Y,Z_female)
    plt.contour(X,Y,lda_male-lda_female,1)
    plt.savefig('lda.pdf')
    plt.show()
                    
    
    #QDA
    qda_male = np.zeros(X.shape)
    qda_female = np.zeros(X.shape)
    cov_male_inv = npl.inv(cov_male)
    cov_female_inv = npl.inv(cov_female)
    plt.figure(2)
    plt.scatter(x_male[:,0],x_male[:,1],color = 'blue')
    plt.scatter(x_female[:,0],x_female[:,1], color = 'red')
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[1]):
            XY = [X[i,j], Y[i,j]]
            Z_male[i,j] = 1/(2*np.pi*np.sqrt(npl.det(cov_male)))*np.exp(-1/2*np.matmul(np.matmul((XY-mu_male),cov_male_inv),(XY-mu_male).T))
            Z_female[i,j] = 1/(2*np.pi*np.sqrt(npl.det(cov_female)))*np.exp(-1/2*np.matmul(np.matmul((XY-mu_female),cov_female_inv),(XY-mu_female).T))
    
            qda_male[i,j] = -1/2*np.matmul(np.matmul((XY-mu_male),cov_male_inv),(XY-mu_male).T) \
                 + np.log(pi_male) - 1/2*np.log(npl.det(cov_male)) 
        
            qda_female[i,j] = -1/2*np.matmul(np.matmul((XY-mu_female),cov_female_inv),(XY-mu_female).T) \
                 + np.log(pi_female) - 1/2*np.log(npl.det(cov_female)) 
                    
    plt.contour(X,Y,Z_male)
    plt.contour(X,Y,Z_female)
    plt.contour(X,Y,qda_male-qda_female,1)
    plt.savefig('qda.pdf')
    plt.show()
                
    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    pi_male = 0.5
    pi_female = 0.5
    #LDA
    
    #QDA

    cov_male_inv = npl.inv(cov_male)
    cov_female_inv = npl.inv(cov_female)
    y_qda = np.zeros(len(y))
    
    cov_inv = npl.inv(cov)
    y_lda = np.zeros(len(y))
    for i in range(0,len(x)):
        XY = x[i,:]
        qda_male = -1/2*np.matmul(np.matmul((XY-mu_male),cov_male_inv),(XY-mu_male).T) \
                 + np.log(pi_male) - 1/2*np.log(npl.det(cov_male)) 
        
        qda_female = -1/2*np.matmul(np.matmul((XY-mu_female),cov_female_inv),(XY-mu_female).T) \
                 + np.log(pi_female) - 1/2*np.log(npl.det(cov_female))
                 
        lda_male = np.matmul(np.matmul(XY,cov_inv),mu_male) - \
                    1/2*np.matmul(np.matmul(mu_male.T,cov_inv),mu_male) + np.log(pi_male)
                    
        lda_female = np.matmul(np.matmul(XY,cov_inv),mu_female) - \
                    1/2*np.matmul(np.matmul(mu_female.T,cov_inv),mu_female) + np.log(pi_female)         
        
        if qda_male > qda_female:
            y_qda[i] = 1
        else:
            y_qda[i] = 2

        if lda_male > lda_female:
            y_lda[i] = 1
        else:
            y_lda[i] = 2
    
    mis_lda = 1-sum(y_lda==y)/len(y)
    mis_qda = 1-sum(y_qda==y)/len(y)

    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    print(mis_LDA)
    print(mis_QDA)
    
    
#    qda_male = -1/2*np.matmul(np.matmul((XY-mu_male),cov_male_inv),(XY-mu_male).T) \
#                + np.log(pi_male) - 1/2*np.log(npl.det(cov_male))
#    
#    qda_female = -1/2*np.matmul(np.matmul((XY-mu_female),cov_female_inv),(XY-mu_female).T) \
#                + np.log(pi_female) - 1/2*np.log(npl.det(cov_female)) 
#                
#    lda_male[i,j] = np.matmul(np.matmul(XY,cov_inv),mu_male) - \
#                    1/2*np.matmul(np.matmul(mu_male.T,cov_inv),mu_male) + np.log(pi_male)
#                    
#    lda_female[i,j] = np.matmul(np.matmul(XY,cov_inv),mu_female) - \
#                    1/2*np.matmul(np.matmul(mu_female.T,cov_inv),mu_female) + np.log(pi_female)            
#            
    
