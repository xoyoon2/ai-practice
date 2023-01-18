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

    def loss(self, X_batch, y_batch=None):
        # print(self.params['W1'].shape, self.params['W2'].shape)
        Hiddens = [X_batch]
        caches = []
        """ Forward pass """
        for i in range(len(self.params) // 2):
            H, cache = affine_relu_forward(
                Hiddens[i], self.params[f"W{i+1}"], self.params[f"b{i+1}"]
            )
            Hiddens.append(H)
            caches.append(cache)

        """ if y is None, test mode: return scores """
        scores = Hiddens[-1]
        if y_batch is None:
            return scores

        """ Backward pass """

        grads = {}
        loss, dscores = softmax_loss(scores, y_batch)
        weight_norm = 0
        for i in range(len(self.params) // 2):
            weight_norm += np.sum(self.params[f"W{i+1}"] ** 2)
        loss += 0.5 * self.reg * weight_norm
        dHs = [dscores]
        for i in range(len(self.params) // 2):
            dH, dW, db = affine_relu_backward(
                dHs[i], caches[len(self.params) // 2 - i - 1]
            )
            dW += self.reg * self.params[f"W{len(self.params) // 2 - i}"]
            (
                grads[f"W{len(self.params) // 2 - i}"],
                grads[f"b{len(self.params) // 2 - i}"],
            ) = (dW, db)
            dHs.append(dH)

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
