import numpy as np
from .base import BaseModel

class PerceptronScratch(BaseModel):
    def __init__(self, standardize: bool = True):
        super().__init__()
        self.standardize = bool(standardize)

    def fit(self, X, y, epochs=100, lr=0.01, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names

        if warm_start and (self.nan_mean is not None):
            Xp = self._prep_transform(X, standardize=self.standardize)
        else:
            Xp = self._prep_fit(X, standardize=self.standardize)

        y = np.asarray(y, dtype=np.int32).reshape(-1)
        n, d = Xp.shape

        if (self.weights is None) or (not warm_start):
            self.weights = np.zeros(d, dtype=np.float64)
            self.bias = 0.0

        # labels {-1,+1}
        ys = np.where(y == 1, 1, -1).astype(np.int32)

        rng = np.random.RandomState(42)
        for _ in range(int(epochs)):
            idx = np.arange(n)
            rng.shuffle(idx)
            for i in idx:
                xi = Xp[i]
                margin = ys[i] * (xi @ self.weights + self.bias)
                if margin <= 0.0:
                    self.weights += float(lr) * ys[i] * xi
                    self.bias += float(lr) * ys[i]

        self.history["trained_epochs"] = self.history.get("trained_epochs", 0) + int(epochs)

    def predict(self, X):
        Xp = self._prep_transform(X, standardize=self.standardize)
        s = Xp @ self.weights + self.bias
        return (s >= 0.0).astype(np.int32)
