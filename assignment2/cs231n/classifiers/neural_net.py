import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model


def relu(x):
  return x if x > 0 else 0

def dRelu(x):
  return 1 if x > 0 else 0

def flat_for(a, f):
  r = a.copy()
  r = r.reshape(-1)
  for i, v in enumerate(r):
    r[i] = f(v)
  return r.reshape(a.shape)


def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  f1 = X.dot(W1) + b1  # (N, H)
  a1 = flat_for(f1, relu)
  scores = a1.dot(W2) + b2  # (N, C)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################
  scores_max = np.max(scores, axis=1)  # (N,)
  scores -= scores_max.reshape(-1, 1)  # (N, C)
  exp_scores = np.exp(scores)
  sums = np.sum(exp_scores, axis=1)  # (N,)
  log_sums = np.log(sums)
  scores_y = np.array([scores[i, y[i]] for i in xrange(X.shape[0])])
  loss = np.sum(-scores_y + log_sums)

  loss /= X.shape[0]
  loss += .5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################

  # input a1, output scores
  dW2 = np.zeros_like(W2)
  dB2 = np.zeros_like(b2)
  for i in xrange(X.shape[0]):  # # of samples
    dW2 += 1./sums[i] * exp_scores[i] * a1[i].reshape(-1, 1)
    dB2 += 1./sums[i] * exp_scores[i]  # for dB2, where there's a1[i], that is 1
    dW2[:, y[i]] -= a1[i]
    dB2[y[i]] -= 1
  dW2 /= X.shape[0]  # # of samples
  dB2 /= X.shape[0]
  dW2 += reg * W2
  # dB's don't have regularization term

  # dB2 = np.zeros_like(b2)
  # for i in xrange(a1.shape[0]):
  #   dB2 += 1./sums[i] * exp_scores[i]
  #   dB2[y[i]] -= 1
  # dB2 /= X.shape[0]

  dW1 = np.zeros_like(W1)
  dB1 = np.zeros_like(b1)
  for i in xrange(X.shape[0]):
    dLi_dFC2i = exp_scores[i] / sums[i]  # (C,)
    dLi_dFC2i[y[i]] -= 1
    dLi_dA1i = W2.dot(dLi_dFC2i)  # (H,)
    dA1i_dFC1i = flat_for(f1[i], dRelu)  # (H,)
    dLi_dFC1i = dLi_dA1i * dA1i_dFC1i
    dW1 += dLi_dFC1i * X[i].reshape(-1, 1)
    dB1 += dLi_dFC1i
  dW1 /= X.shape[0]
  dB1 /= X.shape[0]
  dW1 += reg * W1
  # dB's don't have regularization term

  grads['W1'] = dW1
  grads['b1'] = dB1
  grads['W2'] = dW2
  grads['b2'] = dB2

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads

