import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    ddW = np.zeros(W.shape)
    ddWyi = np.zeros(W[0].shape)
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        ddW[j] = X[:, i] ## be careful, it's a reference
        ddWyi += ddW[j]
    ddW[y[i]] = -ddWyi
    dW += ddW
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * np.sum(W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  import time

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # scores
  wx = W.dot(X)

  # margin
  delta = 1

  # wxy chooses scores of right labels
  # its shape is (#samples,)
  wxy = [ wx[y[i], i] for i in xrange(wx.shape[1]) ]

  # judge expression
  # remember to exclude on y[i]'s
  judge = wx - wxy + delta
  # make judge 0 on y[i]
  for i in xrange(wx.shape[1]):
    judge[y[i], i] = 0

  # mass is a matrix holding all useful temp results
  # shape of judge is (#class, #train)
  mass = np.maximum(0, judge)
  
  loss = np.sum(mass) / X.shape[1]
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # weight to be producted by X
  # its shape is (#classes, #samples)
  weight = np.array((judge > 0).astype(int))

  # weights on y[i] needs special care
  weight_yi = -np.sum(weight, axis=0)
  for i in xrange(wx.shape[1]):
    weight[y[i], i] = weight_yi[i]

  # half vectorized
  # double the speed than non-vectorized
  # still the bottleneck; don't know how to fully vectorize it
#  tic = time.time()
  for i in xrange(X.shape[1]):
    ddW = X[:, i] * weight[:, i].reshape(-1, 1)
    dW += ddW
#  toc = time.time()
#  print 'adding ddW takes %fs' % (toc - tic)

  dW /= X.shape[1]
  dW += reg * np.sum(W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
