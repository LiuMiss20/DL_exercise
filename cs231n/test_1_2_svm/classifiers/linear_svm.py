import numpy as np 
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
	"""
	原始方式 实现 svm loss function 和 gradien

	Input:
		W : A numpy array of shape(D, C) containing weights
		X : A numpy array of shape(N, D) containing a minibatch of data
		y : A numpy array of shape(N, ) containing training labels; 
		reg : (float) regularization strength 正则化强度？
 
	Return:
		loss: float
		gradien with W, same shape as W

	"""

	# 1、初始化 梯度 为 0
	dW = np.zeros(W.shape)
	# 2、计算 总损失 和 梯度
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	for i in range(num_train):
		score = X[i].dot(W)
		s_yi = score[y[i]]
		for j in range(num_classes):
			if j == y[i]:
				continue
			s_j = score[j]
			margin = s_j - s_yi + 1
			if margin > 0:
				loss += margin
				dW[:, y[i]] += -X[i, :]  # 正确 分类的梯度
				dW[:, j] += X[i, :]      # 错误 分类的梯度

	# 3、计算总损失的平均值
	loss /= num_train 

	# 4、加入 正则化损失
	loss += reg * np.sum(W*W)


	return loss, dW

	



def svm_loss_vectorized(W, X, y, reg):
	"""
	向量化 实现 svm loss function

	Input:
		W : A numpy array of shape(D, C) containing weights
		X : A numpy array of shape(N, D) containing a minibatch of data
		y : A numpy array of shape(N, ) containing training labels; 
		reg : (float) regularization strength 正则化强度？
 
	Return:
		loss: float
		gradien with W, same shape as W
	"""

	
	dW = np.zeros(W.shape)
	num_classes = W.shape[1]
	num_train = X.shape[0]

	# 向量化 实现 svm loss
	scores = X.dot(W)

	scores_correct = scores[np.arange(num_train), y]  # 1 by N
	scores_correct = np.reshape(scores_correct, (num_train, -1)) # N by 1

	margins = scores - scores_correct + 1 # N by C
	margins = np.maximum(0, margins)
	margins[np.arange(num_train), y] = 0

	loss = 0.0
	loss += np.sum(margins) / num_train
	loss += reg * np.sum(W * W)




	# 向量化 实现 svm loss 的梯度
	margins[margins > 0] = 1
	row_sum = np.sum(margins, axis=1)  # 1 by N
	margins[np.arange(num_train), y] = -row_sum

	dW += np.dot(X.T, margins) / num_train + reg * W  # D by C




	return loss, dW













