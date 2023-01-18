import numpy as np
from classifiers.loss_grad_svm import *
from classifiers.loss_grad_softmax import *


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
      self,
      X,
      y,
      learning_rate=1e-3,
      reg=1e-5,
      epoch=100,
      batch_size=256,
      verbose=True,
    ):
        loss_history = []
        (num_train, dim) = X.shape
        """ If there are C classes, value of entry of y is one of 0, 1, ..., C-1, indicating each class for each entry """
        """ In digit recognition, y[i] = k means label of i'th image is digit k (=class k) """
        num_classes = np.max(y) + 1
        """ We only use W (no b), hence bias trick should be preprocessed in X, so dim is one bigger than actual dimension of data """
        self.W = np.random.randn(dim, num_classes) * 1e-5

        for it in range(epoch):
            indices = np.random.choice(num_train, batch_size)
            X_batch, y_batch = X[indices], y[indices]
            loss, dW = self.loss(X_batch, y_batch, reg)
            self.W -= learning_rate * dW

            if epoch == 1:
                """Should be approximately (num_classes - 1)"""
                print(f"Initial loss: {loss}")
            if verbose and it % 10 == 0:
                print(f"Iteration: {it}, loss: {loss}")
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        scores = X @ self.W
        y = np.argmax(scores, axis=1)
        return y

    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss(self.W, X_batch, y_batch, reg)
