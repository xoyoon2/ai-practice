import numpy as np
from scipy import stats


class KNN:
    def __init__(self):
        pass

    def train(self, X, Y):
        """
        Remember all train data
        X.shape = (N, D)  (2D, specifically D = 28 * 28)
        Y.shape = (N,)    (1D)
        """
        self.X = X
        self.Y = Y

    def predict(self, X, k):
        num_test = X.shape[0]
        distances = (
            (
                np.sum(self.X**2, axis=1)[:, np.newaxis]
                + np.sum(X**2, axis=1)
                - 2 * np.matmul(self.X, X.T)
            )
        ).T  # Sqrt doesnt change anything
        Y = np.zeros(num_test)
        for n in range(num_test):
            idx = np.argsort(distances[n])[:k]
            candidates = self.Y[idx]
            label = stats.mode(candidates)[0][0]
            Y[n] = label
        return Y
