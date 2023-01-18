import numpy as np

""" dq means dL/dq """


def affine_forward(x, w, b):
    out = x @ w + b
    cache = (x, w, b)
    #print(out.shape)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    dw = x.T @ dout
    db = dout.sum(axis=0)
    dx = dout @ w.T
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    #print('relu_forward:', out.shape)
    return out, cache


def relu_backward(dout, cache):
    x = cache
    mask = np.maximum(x, 0)
    mask[mask > 0] = 1
    dx = dout * mask
    return dx


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def softmax_loss(scores, y):
    num_batch, num_classes = scores.shape[0], scores.shape[1]
    unnormalized_probs = np.exp(scores)
    probs = unnormalized_probs / np.sum(unnormalized_probs, axis=1)[:, np.newaxis]
    losses = -np.log(probs[range(num_batch), y])
    loss = np.sum(losses) / num_batch

    dprobs = np.zeros(probs.shape)
    dprobs[range(num_batch), y] = -1 / probs[range(num_batch), y]
    du = np.ones(unnormalized_probs.shape)
    du = du / np.sum(unnormalized_probs, axis=1)[:, np.newaxis]
    du[range(num_batch), y] -= 1 / unnormalized_probs[range(num_batch), y]
    dscores = du * np.exp(scores) / num_batch
    return loss, dscores
