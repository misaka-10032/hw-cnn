import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # loss
  # shape of scores is (#classes, #samples)

  scores = W.dot(X)
  scores_max = np.max(scores, axis=0)
  scores -= scores_max
  exp_scores = np.exp(scores)
  sums = np.sum(exp_scores, axis=0)
  log_sums = np.log(sums)
  scores_y = np.array([scores[y[i], i] for i in xrange(X.shape[1])])
  loss = np.sum(-scores_y + log_sums)

  loss /= X.shape[1]
  loss += .5 * np.sum(W * W)

  # dW
  for i in xrange(X.shape[1]):
    # dW += 1./sums[i] * log_sums[i] * X[:, i]
    dW += 1./sums[i] * exp_scores[:, i].reshape(-1, 1) * X[:, i]
    dW[y[i]] -= X[:, i]

  dW /= X.shape[1]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss, dW = softmax_loss_naive(W, X, y, reg)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
