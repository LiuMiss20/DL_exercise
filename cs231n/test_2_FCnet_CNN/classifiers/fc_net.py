from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    '''
    2层FCnet with ReLU and softmax loss

    架构为：affine - relu - affine - softmax

    input dimension： D
    hidden dimension : H
    classes : C
    '''
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                weight_scale=1e-3, reg=0.0):
        '''
        初始化 一个新的net

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        '''

        self.params = {}
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros((1, hidden_dim))
        self.params['b2'] = np.zeros((1, num_classes))
        self.reg = reg



    def loss(self, X, y=None):
        '''
        计算 小批次data的 loss 和 gradient

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
        scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.

        '''
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        # print ('here FC net start')
        # print (W1.shape)
        # print (W2.shape)
        # print (b1.shape)
        # print (b2.shape)
        # print ('here FC net stop')

        # 1、 向前传播  affine + relu -->  affine   --> softmax
        relu_out, cache = affine_relu_forward(X, W1, b1)
        scores, scores_cache = affine_forward(relu_out, W2, b2)

        if y is None:
            return scores

        # 2、 向后传播   softmax --> affine -->affine + relu
        grads = {}

        #loss, dscores = svm_loss(scores, y)
        data_loss, dscores = softmax_loss(scores, y)
        loss = data_loss + 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))

        dout, dw2, db2 = affine_backward(dscores, scores_cache)     
        dx, dW1, db1  = affine_relu_backward(dout, cache)

        N = X.shape[0]
        grads['W2'] = dw2 / N + self.reg * W2
        grads['b2'] = db2 
        grads['W1'] = dW1  / N + self.reg * W1
        grads['b1'] = db1 


        return loss, grads


class FullyConnectedNet(object):
    '''
    FC net
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    '''
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
    
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        
        self.use_batchnorm = use_batchnorm
        self.use_dropout  = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        # 这里存储的是每个layer的大小，因为中间的是list，所以要把前后连个加上list来做
        layers_dims = [input_dim] + hidden_dims + [num_classes]
        
        for i in range(self.num_layers):
            self.params['W' + str(i+1)] = weight_scale * np.random.randn(layers_dims[i], layers_dims[i+1])
            self.params['b' + str(i+1)] = np.zeros((1, layers_dims[i+1])) 
            
            if self.use_batchnorm and i < len(hidden_dims): # #最后一层是不需要batchnorm
                self.params['gamma' + str(i+1)] = np.ones((1, layers_dims[i+1]))
                self.params['beta' + str(i+1)] = np.zeros((1, layers_dims[i+1]))
                
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
                
        
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers -1)]  
            
            
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
    
    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'    
    
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode   
                
        h, cache1, cache2, cache3,cache4, bn, out = {}, {}, {}, {}, {}, {},{}        

        # 存储每一层的out，按照逻辑，X就是out0[0]        
        out[0] = X 

        ########################### 计算 scores  ########################## 
        # 非最后一层
        for i in range(self.num_layers - 1):
            
            # 获取每一层的参数
            W, b = self.params['W' + str(i+1)],self.params['b' + str(i + 1)]
            
            if self.use_batchnorm:
                gamma, beta = self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)]
                h[i], cache1[i] = affine_forward(out[i], W, b) 
                bn[i], cache2[i] = batchnorm_forward(h[i], gamma, beta, self.bn_params[i])
                out[i+1], cache3[i] = relu_forward(bn[i])
            else:        
                out[i+1], cache3[i] = affine_relu_forward(out[i], W, b)
        
        # 最后一层
        W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
        scores, cache = affine_forward(out[self.num_layers-1], W, b)
        
        # 判断 mode
        if mode == 'test':   
            return scores       
        
        ########################## 计算 loss   ########################## 
        loss, reg_loss, grads = 0.0, 0.0, {}      
        data_loss, dscores = softmax_loss(scores, y) 
        
        for i in range(self.num_layers):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W' + str(i+1)]*self.params['W' + str(i+1)])
        
        loss = data_loss + reg_loss
    
    
        ########################## 计算 梯度  ########################## 
        dout, dbn, dh = {}, {}, {}
        t = self.num_layers-1
        dout[t], grads['W'+str(t+1)], grads['b'+str(t+1)] = affine_backward(dscores, cache)
        
        for i in range(t):
            if self.use_batchnorm:
                dbn[t-1-i] = relu_backward(dout[t-i], cache3[t-1-i]) 
                dh[t-1-i], grads['gamma'+str(t-i)], grads['beta'+str(t-i)] = batchnorm_backward(dbn[t-1-i], cache2[t-1-i])
                dout[t-1-i], grads['W'+str(t-i)], grads['b'+str(t-i)] = affine_backward(dh[t-1-i], cache1[t-1-i])
            else:    
                dout[t-1-i], grads['W'+str(t-i)], grads['b'+str(t-i)] = affine_relu_backward(dout[t-i], cache3[t-1-i])
                
        for i in range(self.num_layers):        
            grads['W'+str(i+1)] += self.reg * self.params['W' + str(i+1)]

            
        return loss, grads















