import numpy as np
from .base import BaseModel

class LogisticRegressionScratch(BaseModel):
    def __init__(self, l2: float = 0.0, standardize: bool = True):
        super().__init__()
        self.l2 = float(l2)
        self.standardize = bool(standardize)

    def fit(self, X, y, epochs=100, lr=0.01, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names

        if warm_start and (self.nan_mean is not None):
            Xp = self._prep_transform(X, standardize=self.standardize)
        else:
            Xp = self._prep_fit(X, standardize=self.standardize)

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = Xp.shape

        if (self.weights is None) or (not warm_start):
            self.weights = np.zeros(d, dtype=np.float64)
            self.bias = 0.0

        for _ in range(int(epochs)):
            z = Xp @ self.weights + self.bias
            p = self._sigmoid(z)
            grad_w = (Xp.T @ (p - y)) / n + self.l2 * self.weights
            grad_b = float(np.mean(p - y))
            self.weights -= float(lr) * grad_w
            self.bias -= float(lr) * grad_b

        self.history["trained_epochs"] = self.history.get("trained_epochs", 0) + int(epochs)

    def predict(self, X):
        Xp = self._prep_transform(X, standardize=self.standardize)
        p = self._sigmoid(Xp @ self.weights + self.bias)
        return (p >= 0.5).astype(np.int32)

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        Returns a list of probabilities for the positive class (churn).
        Format matches sklearn: array of shape (n_samples, 2)
        """
        Xp = self._prep_transform(X, standardize=self.standardize)
        p = self._sigmoid(Xp @ self.weights + self.bias)
        # Return [prob_0, prob_1] for each sample
        return np.column_stack((1 - p, p))
