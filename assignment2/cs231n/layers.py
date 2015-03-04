import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  X = x.reshape(x.shape[0], -1)  # (N, D)
  out = X.dot(w) + b  # (N, M)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: (M,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  X = x.reshape(x.shape[0], -1)
  dx = dout.dot(w.T).reshape(x.shape)
  dw = X.T.dot(dout)
  db = np.ones(X.shape[0]).dot(dout)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.zeros_like(x)
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  for i, v in np.ndenumerate(x):
    out[i] = x[i] if x[i] > 0 else 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = np.zeros_like(dout), cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  for i, v in np.ndenumerate(dout):
    dx[i] = dout[i] if x[i] > 0 else 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros([N, F, H_out, W_out])
  xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  w_reshape = w.reshape(F, -1).T  # (F, C, HH, WW) -> (D, M) !!!
  for h_idx in xrange(H_out):
    for w_idx in xrange(W_out):
      h_start = h_idx * stride
      h_end = h_start + HH
      w_start = w_idx * stride
      w_end = w_start + WW
      roi = xp[:, :, h_start:h_end, w_start:w_end]  # (N, C, HH, WW)
      out[:, :, h_idx, w_idx], _ = affine_forward(roi, w_reshape, b)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives. (N, F, H_out, W_out)
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x (N, C, H, W)
  - dw: Gradient with respect to w (F, C, HH, WW)
  - db: Gradient with respect to b (F,)
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
  w_reshape = w.reshape(F, -1).T  # remember this T

  dx = np.zeros_like(x)
  dxp = np.zeros_like(xp)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  # dw, db
  for h_idx in xrange(H_out):
    for w_idx in xrange(W_out):
      h_start = h_idx * stride
      h_end = h_start + HH
      w_start = w_idx * stride
      w_end = w_start + WW
      sub_xp = xp[:, :, h_start:h_end, w_start:w_end]
      sub_dout = dout[:, :, h_idx, w_idx].reshape(N, -1)
      cache = sub_xp, w_reshape, b
      ddxp, ddw, ddb = affine_backward(sub_dout, cache)
      # dxp works the magic similarly, though I didn't deduce some formula like that for dw and db
      dxp[:, :, h_start:h_end, w_start:w_end] += ddxp.reshape(N, C, HH, WW)
      dw += ddw.T.reshape(dw.shape)
      db += ddb

  dx = dxp[:, :, pad:-pad, pad:-pad]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  HH = (H - pool_height) / stride + 1
  WW = (W - pool_width) / stride + 1
  out = np.zeros((N, C, HH, WW))
  for h in xrange(HH):
    for w in xrange(WW):
      h_start = h * stride
      h_end = h_start + pool_height
      w_start = w * stride
      w_end = w_start + pool_width
      out[:, :, h, w] = np.max(x[:, :, h_start:h_end, w_start:w_end], (2, 3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives (N, C, HH, WW)
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x (N, C, H, W)
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  HH = (H - pool_height) / stride + 1
  WW = (W - pool_width) / stride + 1
  dx = np.zeros_like(x)

  for h in xrange(HH):
    for w in xrange(WW):
      h_start = h * stride
      h_end = h_start + pool_height
      w_start = w * stride
      w_end = w_start + pool_width
      # TODO: how to do it faster with python? reshape?
      maxes = np.max(x[:, :, h_start:h_end, w_start:w_end], (2, 3))
      for n in xrange(N):
        for c in xrange(C):
          max_nc = maxes[n, c]
          for h_offset in xrange(pool_height):
            for w_offset in xrange(pool_width):
              if x[n, c, h_start+h_offset, w_start+w_offset] == max_nc:
                dx[n, c, h_start+h_offset, w_start+w_offset] = dout[n, c, h, w]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

