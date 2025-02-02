"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod

def hello_linear_classifier():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from linear_classifier.py!')


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier(object):
  """ An abstarct class for the linear classifiers """
  # Note: We will re-use `LinearClassifier' in both SVM and Softmax
  def __init__(self):
    random.seed(0)
    torch.manual_seed(0)
    self.W = None

  def train(self, X_train, y_train, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg,
                  num_iters, batch_size, verbose)
    self.W, loss_history = train_linear_classifier(*train_args)
    return loss_history

  def predict(self, X):
    return predict_linear_classifier(self.W, X)

  @abstractmethod
  def loss(self, W, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
    - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an tensor of the same shape as W
    """
    raise NotImplementedError

  def _loss(self, X_batch, y_batch, reg):
    self.loss(self.W, X_batch, y_batch, reg)

  def save(self, path):
    torch.save({'W': self.W}, path)
    print("Saved in {}".format(path))

  def load(self, path):
    W_dict = torch.load(path, map_location='cpu')
    self.W = W_dict['W']
    print("load checkpoint file: {}".format(path))



class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return softmax_loss_vectorized(W, X_batch, y_batch, reg)



#**************************************************#
################## Section 1: SVM ##################
#**************************************************#

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples. When you implment the regularization over W, please DO NOT
  multiply the regularization term by 1/2 (no coefficient).

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  dW = torch.zeros_like(W) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = W.t().mv(X[i])
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # Compute the gradient of the loss function and store it dW. (part 1)
        # Rather than first computing the loss and then computing the
        # derivative, it is simple to compute the derivative at the same time
        # that the loss is being computed.
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * torch.sum(W * W)

  # Compute the gradient of the loss function and store it in dW. (part 2)    #
  dW /= num_train
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation. When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  loss = 0.0
  dW = torch.zeros_like(W) # initialize the gradient as zero

  # Implement a vectorized version of the structured SVM loss, storing the
  # result in loss.
  num_train = X.shape[0]

  # score matrix: (N, C) = (N, D) * (D, C)
  scores = torch.mm(X, W)

  # correct-class scores: (N,)
  correct_scores = scores[torch.arange(num_train), y]

  # exclude negative and correct class margins
  margin = scores - correct_scores.unsqueeze(1) + 1
  margin[margin <= 0] = 0
  margin[torch.arange(num_train), y] = 0

  # compute the loss including regularization
  loss = torch.sum(margin) / num_train + reg * torch.sum(W * W)

  # Implement a vectorized version of the gradient for the structured SVM
  # loss, storing the result in dW.
  # Hint: Instead of computing the gradient from scratch, it may be easier to
  # reuse some of the intermediate values that you used to compute the loss.                                                                     #
  margin[margin > 0] = 1
  margin[torch.arange(num_train), y] -= margin.sum(dim=1)

  # compute gradient: (D, C) = (D, N) * (N, C)
  dW = X.t().mm(margin) / num_train
  dW += 2 * reg * W

  return loss, dW


def sample_batch(X, y, num_train, batch_size):
  """
  Sample batch_size elements from the training data and their
  corresponding labels to use in this round of gradient descent.
  """
  X_batch = None
  y_batch = None
  # TODO: Store the data in X_batch and their corresponding labels in
  # y_batch; after sampling, X_batch should have shape (batch_size, dim)
  # and y_batch should have shape (batch_size,)
  # Hint: Use torch.randint to generate indices.
  indices = torch.randint(0, num_train, (batch_size,))
  X_batch = X[indices, :]
  y_batch = y[indices]

  return X_batch, y_batch


def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3,
                            reg=1e-5, num_iters=100, batch_size=200,
                            verbose=False):
  """
  Train this linear classifier using stochastic gradient descent.

  Inputs:
  - loss_func: loss function to use when training. It should take W, X, y
    and reg as input, and output a tuple of (loss, dW)
  - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
    classifier. If W is None then it will be initialized here.
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
    means that X[i] has label 0 <= c < C for C classes.
  - learning_rate: (float) learning rate for optimization.
  - reg: (float) regularization strength.
  - num_iters: (integer) number of steps to take when optimizing
  - batch_size: (integer) number of training examples to use at each step.
  - verbose: (boolean) If true, print progress during optimization.

  Returns: A tuple of:
  - W: The final value of the weight matrix and the end of optimization
  - loss_history: A list of Python scalars giving the values of the loss at each
    training iteration.
  """
  # assume y takes values 0...K-1 where K is number of classes
  num_train, dim = X.shape
  if W is None:
    # lazily initialize W
    num_classes = torch.max(y) + 1
    W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
  else:
    num_classes = W.shape[1]

  # Run stochastic gradient descent to optimize W
  loss_history = []
  for it in range(num_iters):
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # evaluate loss and gradient
    loss, grad = loss_func(W, X_batch, y_batch, reg)
    loss_history.append(loss.item())

    # perform parameter update using the gradient and the learning rate
    W -= learning_rate * grad

    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss))

  return W, loss_history


def predict_linear_classifier(W, X):
  """
  Use the trained weights of this linear classifier to predict labels for
  data points.

  Inputs:
  - W: A PyTorch tensor of shape (D, C), containing weights of a model
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.

  Returns:
  - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
    elemment of X. Each element of y_pred should be between 0 and C - 1.
  """
  y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
  scores = X.mm(W)
  y_pred = torch.argmax(scores, dim=1)
  return y_pred


def svm_get_search_params():
  """
  Return candidate hyperparameters for the SVM model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """

  learning_rates = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
  regularization_strengths = [1e0, 5e-1, 1e-1, 5e-2]
  return learning_rates, regularization_strengths


def test_one_param_set(cls, data_dict, lr, reg, num_iters=2000):
  """
  Train a single LinearClassifier instance and return the learned instance
  with train/val accuracy.

  Inputs:
  - cls (LinearClassifier): a newly-created LinearClassifier instance.
                            Train/Validation should perform over this instance
  - data_dict (dict): a dictionary that includes
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      as the keys for training a classifier
  - lr (float): learning rate parameter for training a SVM instance.
  - reg (float): a regularization weight for training a SVM instance.
  - num_iters (int, optional): a number of iterations to train

  Returns:
  - cls (LinearClassifier): a trained LinearClassifier instances with
                            (['X_train', 'y_train'], lr, reg)
                            for num_iter times.
  - train_acc (float): training accuracy of the svm_model
  - val_acc (float): validation accuracy of the svm_model
  """
  train_acc = 0.0 # The accuracy is simply the fraction of data points
  val_acc = 0.0   # that are correctly classified.
  
  # train a linear SVM on the training set, compute its accuracy 
  # on the training and validation sets
  X_train, y_train = data_dict['X_train'], data_dict['y_train']
  X_val, y_val = data_dict['X_val'], data_dict['y_val']
  num_train = X_train.shape[0]
  num_val = X_val.shape[0]

  cls.train(X_train, y_train, lr, reg, num_iters)
  y_train_pred = cls.predict(X_train)
  train_acc = torch.sum(y_train == y_train_pred) / num_train

  y_val_pred = cls.predict(X_val)
  val_acc = torch.sum(y_val == y_val_pred) / num_val

  return cls, train_acc, val_acc



#**************************************************#
################ Section 2: Softmax ################
#**************************************************#

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops).  When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an tensor of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  # TODO: Compute the softmax loss and its gradient using explicit loops.
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    score = torch.matmul(X[i], W)
    score -= score.max()

    # softmax probabilities
    exp_score = torch.exp(score)
    total = exp_score.sum()
    softmax = exp_score / total

    # loss = -log(prob of correct class)
    loss -= torch.log(softmax[y[i]])

    # compute gradient
    for j in range(num_class):
      if j == y[i]:
        dW[:, j] += (softmax[j] - 1) * X[i]
      else:
        dW[:, j] += softmax[j] * X[i]
  
  loss /= num_train
  loss += reg * torch.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.  When you implment the
  regularization over W, please DO NOT multiply the regularization term by 1/2
  (no coefficient).

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  # TODO: Compute the softmax loss and its gradient using no explicit loops.
  # Store the loss in loss and the gradient in dW. If you are not careful
  # here, it is easy to run into numeric instability (Check Numeric Stability in
  # https://cs231n.github.io/linear-classify/). Don't forget the regularization!                                                           #
  num_train = X.shape[0]
  
  # scores: (N, C) = (N, D) * (D, C)
  scores = torch.mm(X, W)
  scores -= torch.amax(scores, dim=1, keepdim=True)
  exp_scores = torch.exp(scores)
  exp_total = exp_scores.sum(dim=1, keepdim=True)
  softmax = exp_scores / exp_total
  
  correct_class = softmax[torch.arange(num_train), y]
  loss = -torch.log(correct_class).sum()
  loss = loss / num_train + reg * torch.sum(W * W)

  softmax[torch.arange(num_train), y] -= 1
  dW = X.t().mm(softmax)
  dW = dW / num_train + 2 * reg * W

  return loss, dW


def softmax_get_search_params():
  """
  Return candidate hyperparameters for the Softmax model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """
  learning_rates = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
  regularization_strengths = [1e0, 5e-1, 1e-1, 5e-2, 1e-2]
  return learning_rates, regularization_strengths
