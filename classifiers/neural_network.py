import numpy as np
from layers import *


class NeuralNetwork:
    def __init__(self, input_dim, num_classes, hidden_dims, initial_weight=1e-3, reg=0):
        """
        Initialize network parameters:
        self.params[W1], self.params[W2], ... for weight matrices,
        self.params[b1], self.params[b2], ... for bias vectors.
        """
        self.params = {}
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            self.params[f"W{i+1}"] = initial_weight * np.random.randn(
                dims[i], dims[i + 1]
            )
            self.params[f"b{i+1}"] = initial_weight * np.random.randn(dims[i + 1])
        self.reg = reg
        self.num_layers = len(hidden_dims) + 1

    def loss(self, X, y=None):

        Hiddens = [X]
        caches = []

        """ Forward pass """
        for i in range(1, self.num_layers + 1):
            if i == self.num_layers:
                H, cache = affine_forward(
                    Hiddens[i - 1], self.params[f"W{i}"], self.params[f"b{i}"]
                )
            else:
                H, cache = affine_relu_forward(
                    Hiddens[i - 1], self.params[f"W{i}"], self.params[f"b{i}"]
                )
            Hiddens.append(H)
            caches.append(cache)

        scores = Hiddens[-1]

        """ if y is None, test mode: return scores """
        if y is None:
            return scores

        """ Backward pass """

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)

        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W * W)

        for i in range(self.num_layers, 0, -1):
            if i == self.num_layers:
                dH, dW, db = affine_backward(dscores, caches[i - 1])
            else:
                dH, dW, db = affine_relu_backward(dout, caches[i - 1])
            dW += self.reg * self.params[f"W{i}"]
            grads[f"W{i}"] = dW
            grads[f"b{i}"] = db
            dout = dH

        return loss, grads

    def train(self, data, learning_rate, epoch, batch_size=256, verbose=True):
        train_X, train_y, val_X, val_y = (
            data["train_X"],
            data["train_y"],
            data["val_X"],
            data["val_y"],
        )
        num_train = train_X.shape[0]
        for it in range(epoch):
            indices = np.random.choice(num_train, batch_size)
            X_batch, y_batch = train_X[indices], train_y[indices]
            loss, grads = self.loss(X_batch, y_batch)

            for param in grads:
                self.params[param] -= learning_rate * grads[param]

            if epoch == 1:
                """Should be approximately (num_classes - 1)"""
                print(f"Initial loss: {loss}")
            if verbose and it % 10 == 0:
                val_loss, _ = self.loss(val_X, val_y)
                print(
                    f"Iteration: {it}, train loss: {loss}, validation loss: {val_loss}"
                )

    def predict(self, X):
        scores = self.loss(X)
        y = np.argmax(scores, axis=1)
        return y
