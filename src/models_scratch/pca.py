import numpy as np

class PCAScratch:
    def __init__(self, n_components=10):
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None  # shape (k, d)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # SVD
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]  # (k, d)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
