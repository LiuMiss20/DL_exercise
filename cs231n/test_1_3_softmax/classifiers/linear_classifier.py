#coding:utf-8
import numpy as np 
#from classifiers.linear_svm import *
from classifiers.softmax import *
from past.builtins import xrange

class LinearClassifier(object):
  def __init__(self,W=None):
    self.W = W

  def train(self, X, y,learning_rate=1e-3, reg=1e-5,num_iters=100, batch_size=200, verbose=False):
    """
    使用SGD 训练 线性分类器

    Input：
      - reg：float 正则化强度
      - verbose: boolean : if True, print propress during optimization

    Output:
      - list, the value of the loss function at each training iteration 

    """

    num_train, dim = X.shape
    num_classes = np.max(y) + 1

    # 如果 W 为空，则 随机初始化 最小值
    if self.W is None:
      self.W = np.random.randn(dim, num_classes) * 0.001

    # SGD 优化器
    loss_history = []
    for it in range(num_iters):

      # 1、抽取 batch size 的 example 
      X_batch = None 
      y_batch = None 
       
      idx = np.random.choice(num_train, batch_size, replace=False)
      X_batch = X[idx, :] # (dim, batch_size)
      y_batch = y[idx] # (batch_size,)



      # 2、评估 loss 和 gradient
      loss, grad = self.loss( X_batch, y_batch, reg)
      loss_history.append(loss)

      # 3、更新 参数
      self.W += - grad * learning_rate

      # 4、是否 打印进度条
      if verbose and it % 1 == 0:
        print('iteration %d / %d: loss %f'  % (it, num_iters, loss))

    return loss_history




  def predict(self, X):
    """
    使用 训练好的W 预测 label

    """

    # 1、初始化 预测值 为0
    y_pred = np.zeros(X.shape[0])

    # 2、预测
    score = X.dot(self.W)
    y_pred = np.argmax(score, axis=1)

    return y_pred




  def loss(self, X_batch, y_batch, reg):
    """
    计算 损失 和 导数

    """
    pass

#class LinearSVM(LinearClassifier):
#  def loss(self, X_batch, y_batch, reg):
#    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class LinearSoftmax(LinearClassifier):
  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

