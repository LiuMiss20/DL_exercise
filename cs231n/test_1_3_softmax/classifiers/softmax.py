import numpy as np 
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
	"""
	原始方式 实现 softmax function 和 gradien

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

	# 计算 损失
	for i in range(num_train):
		# 评分函数
		score = X[i].dot(W)

		# 对数概率
		score = np.exp(score)

		# 归一化
		sum_scores = 0.0
		for j in range(num_classes):
			sum_scores += np.sum(score[j])
		score_y_i = score[y[i]] / sum_scores

		# 计算 数据损失 
		loss_i = - np.log(score_y_i)

		# 加入 正则化损失
		loss_i += reg * np.sum(W*W)

		# 计算 梯度
		if loss_i > 0:
			for m in range(num_classes):
				dW[:, y[i]] += -X[i, :]  # 正确 分类的梯度
				dW[:, m] += X[i, :]      # 错误 分类的梯度

		# 计算 总的损失
		loss += loss_i


	# 计算 平均值
	loss /= num_train 
	dW /= num_train

	return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
	"""
	原始方式 实现 softmax function 和 gradien

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

	# 计算 损失
	# 评分函数
	score = X.dot(W)

	# 对数概率
	score = np.exp(score)
	#score = normalized(score)

	# 归一化
	sum_scores = np.sum(score, axis = 1)
	sum_scores = 1.0 / sum_scores
	norm_score = score.T * sum_scores.T
	norm_score = norm_score.T
	
	
	# 计算 损失 和 梯度 
	margins = -np.log(norm_score) # 每个类别的loss
	ve_sum = np.sum(margins,axis=1)/num_classes
	y_trueClass = np.zeros_like(margins)
	y_trueClass[range(num_train), y] = 1.0

	loss += np.sum(ve_sum)  
	loss = loss / num_train  # 计算 均值
	loss = loss + reg * np.sum(W*W) # 加入 正则化 损失
	
	dW += np.dot(X.T,norm_score-y_trueClass) / num_train
	


################
	'''
	print ('here')
	margins = -np.log(norm_score)
	ve_sum = np.sum(margins,axis=1)/num_classes
	y_trueClass = np.zeros_like(margins)
	y_trueClass[range(num_train), y] = 1.0
	loss += (np.sum(ve_sum) / num_train)
	dW += np.dot(X.T,norm_score-y_trueClass)/num_train
	'''

####################

	return loss, dW


def normalized(a):
	sum_scores =  np.sum(a,axis=1)
	sum_scores =  1 / sum_scores
	result = a.T * sum_scores.T
	return result.T
