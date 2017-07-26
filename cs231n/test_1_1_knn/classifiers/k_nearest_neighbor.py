###########  KNN
import numpy as np
from past.builtins import xrange

class KNearestNeighbor(object):
    def __init__(self):
        pass
    
    # 记住 训练数据
    def train(self, X, y):
        """
        X: N x D, 每一行为一个example
        y: N x 1dim
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k=1, num_loops=0):
        """
        X: N x D, 每一行为一个 需要预测label的example
        
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype = self.y_tr.dtype)
        
        # 对每个test image：找出 最近的训练image，预测 最近image的 label
        for i in xrange(num_test):
            distances = np.sum( np.abs( self.X_tr - X[i,:]), axis = 1 )
            min_index = np.argmin(distances)
            y_pred[i] = self.y_tr[min_idex]
        
        return y_pred
        """
        
        if num_loops == 0 :
        	dists = self.compute_distances_no_loops(X)
        elif num_loops == 1 :
        	dists = self.compute_distances_one_loops(X)
        elif num_loops == 2 :
        	dists = self.compute_distances_two_loops(X)
        else:
        	raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

          
    def compute_distances_two_loops(self, X):
        """
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.        
        
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.   
          store the result in dists[i, j]    
        
        """
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in xrange(num_test):
        	for j in xrange(num_train):
        		dists[i,j] = np.sqrt(np.sum(np.square(X[i,:] - self.X_train[j,:])))
        
        return dists



    
    def compute_distances_one_loops(self, X):
        """
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.        
        
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.     
          store the result in dists[i, :].  
        
        """
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in xrange(num_test):
        	#dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train)))
        	a = X[i, :] - self.X_train
        	dists[i, :] = np.sqrt(np.sum(a**2))
        
        return dists

    
    
    # L1 distance
    def compute_distances_no_loops(self, X):
        """
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.        
        
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.       
        
        """
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        M = np.dot(X, self.X_train.T)
        te = np.square(X).sum(axis = 1)
        tr = np.square(self.X_train).sum(axis = 1)
        dists = np.sqrt(-2*M + tr + np.matrix(te).T)
        dists = np.array(dists)

        return dists




    def predict_labels(self, dists, k=1):
	    """
	    Given a matrix of distances between test points and training points,
	    predict a label for each test point.

	    Inputs:
	    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	      gives the distance betwen the ith test point and the jth training point.

	    Returns:
	    - y: A numpy array of shape (num_test,) containing predicted labels for the
	      test data, where y[i] is the predicted label for the test point X[i].  
	    """        	
	    num_test = dists.shape[0]
	    y_pred = np.zeros(num_test)

	    for i in xrange(num_test):
	    	closest_y = []
	    	idx = np.argsort(dists[i,:], -1)
	    	closest_y = self.y_train[idx[:k]] 
	    	closest_set = set(closest_y) #find max label

	    	for idx, item in enumerate(closest_set):
	    		y_pred[i] = item
	    		if idx == 0:
	    			break

	    return y_pred
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    