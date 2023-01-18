import numpy as np


def svm_loss(W, X_batch, y_batch, reg):
    num_batch, num_classes = X_batch.shape[0], W.shape[1]
    scores = X_batch @ W  # num_batch × num_classes
    losses = np.maximum(
        scores - scores[range(num_batch), y_batch][:, np.newaxis] + 1, 0
    )  # advanced indexing & broadcasting
    loss = np.sum(losses) / num_batch - 1 + reg * np.sum(W * W)

    indicator = losses
    indicator[indicator > 0] = 1
    dscores = indicator
    for i in range(num_batch):
        dscores[i][y_batch[i]] -= sum(dscores[i])
    dW = X_batch.T @ dscores / num_batch + (2 * reg * W)
    return loss, dW
