"""
Simple Linear Regression — From Scratch (NumPy only)

Fits y = w*x + b using gradient descent on MSE loss.
Uses a SINGLE feature (the best one) to predict the target.

Course requirement: "functions built from scratch"
"""

import numpy as np
from src.models_scratch.base import BaseModel


class LinearRegressionScratch(BaseModel):
    """
    Simple Linear Regression using gradient descent.

    Only uses ONE feature (selected by highest correlation with target).
    This is the simplest form: y = w*x + b
    """

    def __init__(self):
        super().__init__()
        self.best_feature_idx = None
        self.best_feature_name = None
        # Scaling params (stored so predict() can reuse them)
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _select_best_feature(self, X: np.ndarray, y: np.ndarray) -> int:
        """Pick the single feature with highest |correlation| to y."""
        n_features = X.shape[1]
        correlations = np.array([
            np.corrcoef(X[:, j], y)[0, 1] if np.std(X[:, j]) > 0 else 0.0
            for j in range(n_features)
        ])
        best = int(np.argmax(np.abs(correlations)))
        print(f"   Best feature index: {best}  (corr = {correlations[best]:.4f})")
        return best

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.01,
        feature_names: list = None,
        warm_start: bool = False,
    ) -> None:
        """
        Train simple linear regression via gradient descent.

        Parameters
        ----------
        X : (n_samples, n_features)  — we auto-select the best single feature
        y : (n_samples,)
        epochs : number of gradient-descent iterations
        lr : learning rate
        warm_start : if True and weights exist, continue from previous training
        feature_names : optional list of column names for logging
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # 1. Select best feature
        self.best_feature_idx = self._select_best_feature(X, y)
        if feature_names is not None:
            self.best_feature_name = feature_names[self.best_feature_idx]
            print(f"   Using feature: '{self.best_feature_name}'")

        x = X[:, self.best_feature_idx].copy()  # shape (n,)

        # 2. Z-score normalisation (speeds up convergence)
        self._x_mean = x.mean()
        self._x_std = x.std() if x.std() > 0 else 1.0
        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1.0

        x_norm = (x - self._x_mean) / self._x_std
        y_norm = (y - self._y_mean) / self._y_std

        # 3. Initialise parameters (with warm-start support)
        n = len(x_norm)
        if warm_start and self.weights is not None and self.bias is not None:
            # Continue from existing weights (convert back to normalized space)
            w = self.weights[0] * self._x_std / self._y_std
            b = (self.bias - self._y_mean) / self._y_std + w * self._x_mean / self._x_std
            print(f"   ♻️  Warm start: continuing from existing weights (w={self.weights[0]:.4f}, b={self.bias:.4f})")
        else:
            # Start from scratch
            w = 0.0
            b = 0.0
            print(f"   🆕 Cold start: initializing weights to zero")

        self.history["loss"] = []

        # 4. Gradient descent
        for epoch in range(epochs):
            y_hat = w * x_norm + b
            error = y_hat - y_norm

            # MSE loss
            loss = float(np.mean(error ** 2))
            self.history["loss"].append(loss)

            # Gradients
            dw = (2.0 / n) * np.dot(x_norm, error)
            db = (2.0 / n) * np.sum(error)

            w -= lr * dw
            b -= lr * db

            if epoch % max(1, epochs // 5) == 0:
                print(f"      Epoch {epoch:>5d}/{epochs}  MSE(norm)={loss:.6f}")

        # 5. Convert back to original scale
        #    y = y_std * (w * (x - x_mean)/x_std + b) + y_mean
        #    y = (y_std * w / x_std) * x + (y_std * (b - w * x_mean / x_std) + y_mean)
        self.weights = np.array([self._y_std * w / self._x_std])
        self.bias = float(self._y_std * (b - w * self._x_mean / self._x_std) + self._y_mean)

        print(f"   ✅ Training complete — w={self.weights[0]:.6f}, b={self.bias:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the learned w and b on the selected feature."""
        X = np.asarray(X, dtype=np.float64)
        x = X[:, self.best_feature_idx]
        return self.weights[0] * x + self.bias

    # ------------------------------------------------------------------
    # readable summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        feat = self.best_feature_name or f"feature[{self.best_feature_idx}]"
        return (
            f"Simple Linear Regression (from scratch)\n"
            f"  y = {self.weights[0]:.4f} * {feat} + {self.bias:.4f}\n"
            f"  Trained for {len(self.history.get('loss', []))} epochs"
        )
