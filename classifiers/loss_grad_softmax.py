import numpy as np


def softmax_loss(W, X_batch, y_batch, reg):
    num_batch, num_classes = X_batch.shape[0], W.shape[1]
    scores = X_batch @ W  # num_batch × num_classes
    unnormalized_probs = np.exp(scores)
    probs = unnormalized_probs / np.sum(unnormalized_probs, axis=1)[:, np.newaxis]
    losses = -np.log(probs[range(num_batch), y_batch])
    loss = np.sum(losses) / num_batch + reg * np.sum(W * W)

    dprobs = np.zeros(probs.shape)
    dprobs[range(num_batch), y_batch] = -1/probs[range(num_batch), y_batch]
    du = np.ones(unnormalized_probs.shape)
    du = du / np.sum(unnormalized_probs, axis=1)[:, np.newaxis]
    du[range(num_batch), y_batch] -= 1/unnormalized_probs[range(num_batch), y_batch]
    dscores = du * np.exp(scores)
    dW = X_batch.T @ dscores / num_batch + (2 * reg * W)
    return loss, dW
