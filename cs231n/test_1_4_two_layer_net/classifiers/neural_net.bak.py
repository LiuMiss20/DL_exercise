import numpy as np 
from past.builtins import xrange

class TwoLayerNet(object):
	def __init__(self, input_size, hidden_size, output_size, std=1e-4):
		'''
		初始化model.
		Weights： 初始化为 随机小的值
		biases ： 初始化为 0
		params ： 字典，存储 2层的 Weights，biases
		'''
		self.params = {}
		self.params['W1'] = std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	def loss(self, X, y=None, reg=0.0):
		'''
		计算 net 的 损失和梯度
		'''

		num_example, input_dim = X.shape
		#num_classes = y[1]
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		print ('计算前的b2',b2)
		# 1、计算 类别分数
		f = lambda x : 1.0/(1.0 + np.exp(-x)) # sigmoid 
		#f = lambda x : np.maximum(0, x)  # ReLU
		h1 = np.dot(X, W1) + b1
		f_h1 = f(h1)
		scores = np.dot(f_h1, W2) + b2
		'''
		print ('\n,W1', W1)
		print ('b1', b1)
		print ('h1 = np.dot(X, W1) + b1 \n', h1)


		print ('\n f_h1=f = lambda x : 1.0/(1.0 + np.exp(-x)) \n', f_h1)

		print ('\n f_h1', f_h1)
		print ('W2', W2)
		print ('b2', b2)

		print ('f_h1',f_h1.shape)
		print ('W2',W2.shape)
		print ('scores',scores.shape)
		print ('scores = np.dot(f_h1, W2) + b2 \n', h1)
		'''

		# 如果 没有给定 targets ，则跳过，并返回scores
		if y is None:
			return scores


		# 2、计算 损失 Loss
		y_true = np.zeros(scores.shape)
		y_true[range(num_example), y] = 1.0
		#print ('y_true',y_true)

		exp_scores = np.exp(scores)  # (N,C)
		#print ('exp_scores',exp_scores)

		distance = - np.sum(y_true * exp_scores,axis=1) # (N,)  # cross entropy  # 每个样本的 loss
		print ('distance',distance.shape,distance)

		loss_data_mean = np.sum(distance) / num_example# (1,)    # 所有样本的平均data loss
		print ('loss_data_mean',loss_data_mean)
		



		# 2.4 计算 正则化损失
		RW2 = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) 
		print ('\n RW2 loss',RW2)

		# 2.5 计算 总的损失
		loss = loss_data_mean #+ RW2
		print ('\n 总的损失:total loss',loss)

	

		# 3、计算 参数的梯度
		# 单个样本的 不同类别的 损失 
		dscores = - y_true * exp_scores  # ddistance
		#dscores = - np.log(exp_f_j / np.reshape(exp_sum,(num_example,1)))  + RW2/num_example  # (N,C)

		dW2 = np.dot(f_h1.T , dscores)   # (H,C)   # (N, H).T * (N,C)   
		db2 = np.ones(b2.shape) * dscores # (C,)  = (C,) * (N,C) 
		df_h1 = np.dot(W2 , dscores.T)  # (H,N) 


		dh1 = f_h1 * (1 - f_h1) * df_h1.T  # sigmoid 

		dW1 = np.dot(X.T , dh1)  # (I,H) 
		db1 = np.ones(b1.shape) # (H,) = (H,1)*(H,C)

		print ('\n 梯度dW1',dW1)
		print ('\n 梯度dW2',dW2)
		print ('\n 梯度db1',db1)
		print ('\n 梯度db2',db2)
		print ('\n 计算后的b2',b2)
		


		# 4、更新 参数
		#dW2 += dW2
		#dW1 += dW1

		#db2 += db2
		#db1 += db1

		#print ('\n 更新hou梯度dW1',dW1)
		#print ('\n 更新hou梯度dW2',dW2)
		#print ('\n 更新hou梯度db1',db1)
		#print ('\n 更新hou梯度db2',db2)
		
		grads = {}
		
		grads['W1'] = dW1
		grads['b1'] = db1
		grads['W2'] = dW2
		grads['b2'] = db2

		
		return loss, grads


	def train(self, X, y, X_val, y_val,
				learning_rate=1e3, learning_rate_decay=0.95,
				reg=5e-6, num_iters=3,
				batch_size=5, verbose=False):
		'''
		使用 SGD 训练 net

		'''

		num_train = X.shape[0]
		iterations_per_epoch = max(num_train / batch_size, 1)
		#output_size = np.max(y) + 1
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		loss_history = []
		train_acc_history = []
		val_acc_history = []
		for it in xrange(num_iters):
			print ('\n \n \n##################',it,'###################')

			# 1、抽取 batch size 的 example 
			X_batch = []
			y_batch = []


			idx = np.random.choice(num_train, batch_size, replace=False)
			X_batch = X[idx, :] 
			y_batch = y[idx]
			print ('X_batch',X_batch)
			print ('y_batch',y_batch)

			# 2、评估 loss 和 gradient
			loss, grads = self.loss(X_batch, y=y_batch,  reg=reg)
			loss_history.append(loss)
			print('loss_history',loss_history[-1])

			# 3、更新 参数
			W1 += - grads['W1'] * learning_rate_decay
			W2 += - grads['W2'] * learning_rate_decay
			b1 += - grads['b1'] * learning_rate_decay
			b2 += - grads['b2'] * learning_rate_decay

			# 4、是否 打印进度条
			if verbose and it % 100 == 0:
				print('iteration %d / %d: loss %f' % (it, num_iters, loss))

			# 5、计算 准确率
			if it % iterations_per_epoch == 0:
				# Check accuracy
				train_acc = (self.predict(X_batch) == y_batch).mean()
				val_acc = (self.predict(X_val) == y_val).mean()
				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)

				# Decay learning rate
				learning_rate *= learning_rate_decay

		return {
			'loss_history': loss_history,
			'train_acc_history': train_acc_history,
			'val_acc_history': val_acc_history,
		}


	def predict(self, X):
		# 1、初始化 预测值 为0
		y_pred = np.zeros(X.shape[0])

		# 2、预测
		scores = self.loss(X)
		y_pred = np.argmax(scores, axis=1)

		return y_pred
		





