import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. 
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. 
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None

    z2 = X.dot(W1) + b1
    a2 = np.zeros_like(z2)
    a2 = np.maximum(z2, 0)
    scores = a2.dot(W2) + b2

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores


    # Compute the loss
    loss = None

    exp_scores = np.exp(scores)
    row_sum = exp_scores.sum(axis=1).reshape((N, 1))
    norm_scores = exp_scores / row_sum
    data_loss = -1.0/N * np.log(norm_scores[np.arange(N), y]).sum()
    reg_loss = 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
    loss = data_loss + reg_loss


    # Backward pass: compute gradients
    grads = {}

    delta3 = np.zeros_like(norm_scores)    #delta3 = dloss / dz3
    delta3[np.arange(N), y] -= 1
    delta3 += norm_scores
    grads['W2'] = a2.T.dot(delta3) / N + reg * W2
    #grads['b2'] = np.ones((1,N)).dot(delta3) / N
    grads['b2'] = np.ones(N).dot(delta3) / N

    da2_dz2 = np.zeros_like(z2)
    da2_dz2[z2>0] = 1
    delta2 = delta3.dot(W2.T) * da2_dz2
    grads['W1'] = X.T.dot(delta2) / N + reg * W1
    grads['b1'] = np.ones(N).dot(delta2) / N


    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    self.hyper_params = {}
    self.hyper_params['learning_rate'] = learning_rate
    self.hyper_params['reg'] = reg
    self.hyper_params['batch_size'] = batch_size
    self.hyper_params['hidden_size'] = self.params['W1'].shape[1]
    self.hyper_params['num_iter'] = num_iters

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None


      batch_inx = np.random.choice(num_train, batch_size)
      X_batch = X[batch_inx,:]
      y_batch = y[batch_inx]


      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)


      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']


      if verbose and it % 100 == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
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

    y_pred = None

    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)

    return y_pred