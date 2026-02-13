"""
Multiple Linear Regression — From Scratch (NumPy only)

Fits y = X @ W + b using gradient descent on MSE loss.
Uses ALL features simultaneously.

Course requirement: "functions built from scratch"
"""

import numpy as np
from src.models_scratch.base import BaseModel


class MultipleRegressionScratch(BaseModel):
    """
    Multiple Linear Regression using gradient descent.

    Uses ALL features: y = X @ W + b   (W is a vector of weights)
    """

    def __init__(self):
        super().__init__()
        # Scaling params
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

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
        Train multiple linear regression via gradient descent.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        epochs : number of gradient-descent iterations
        lr : learning rate
        warm_start : if True and weights exist, continue from previous training
        feature_names : optional list of column names for logging
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape

        # 1. Z-score normalisation
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std == 0] = 1.0  # avoid div-by-zero for constant cols

        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1.0

        X_norm = (X - self._X_mean) / self._X_std
        y_norm = (y - self._y_mean) / self._y_std

        # 2. Initialise parameters (with warm-start support)
        if warm_start and self.weights is not None and self.bias is not None:
            # Check dimension compatibility (features may have changed)
            if len(self.weights) == d:
                # Continue from existing weights (convert back to normalized space)
                W = self.weights * self._X_std / self._y_std
                b = (self.bias - self._y_mean) / self._y_std + np.sum(W * self._X_mean / self._X_std)
                print(f"   ♻️  Warm start: continuing from existing weights ({d} features)")
            else:
                # Dimension mismatch - fall back to cold start
                W = np.zeros(d)
                b = 0.0
                print(f"   ⚠️  Dimension mismatch! Old model has {len(self.weights)} weights, need {d}. Cold start.")
        else:
            # Start from scratch
            W = np.zeros(d)
            b = 0.0
            print(f"   🆕 Cold start: initializing {d} weights to zero")

        self.history["loss"] = []

        # 3. Gradient descent
        for epoch in range(epochs):
            y_hat = X_norm @ W + b
            error = y_hat - y_norm

            # MSE loss
            loss = float(np.mean(error ** 2))
            self.history["loss"].append(loss)

            # Gradients
            dW = (2.0 / n) * (X_norm.T @ error)
            db = (2.0 / n) * np.sum(error)

            W -= lr * dW
            b -= lr * db

            if epoch % max(1, epochs // 5) == 0:
                print(f"      Epoch {epoch:>5d}/{epochs}  MSE(norm)={loss:.6f}")

        # 4. Convert normalised weights → original scale
        #    y = y_std * (X_norm @ W + b) + y_mean
        #    y = y_std * ((X - X_mean)/X_std @ W + b) + y_mean
        #    y = sum_j [ (y_std * W_j / X_std_j) * X_j ] + (y_std * (b - sum_j W_j * X_mean_j / X_std_j) + y_mean)
        self.weights = self._y_std * W / self._X_std
        self.bias = float(
            self._y_std * (b - np.sum(W * self._X_mean / self._X_std)) + self._y_mean
        )

        print(f"   ✅ Training complete — {d} weights learned")
        if feature_names is not None:
            top3 = np.argsort(np.abs(self.weights))[::-1][:3]
            for rank, idx in enumerate(top3, 1):
                print(f"      Top-{rank} weight: {feature_names[idx]} = {self.weights[idx]:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the learned W and b."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias

    # ------------------------------------------------------------------
    # Closed-form solution (Normal Equation) for comparison
    # ------------------------------------------------------------------
    def fit_closed_form(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None,
    ) -> None:
        """
        Solve using the Normal Equation: W = (X^T X)^{-1} X^T y
        No epochs, no learning rate — exact solution.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Add bias column
        X_b = self._add_bias(X)  # (n, d+1)

        # Normal equation
        W_full = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y)

        self.bias = float(W_full[0])
        self.weights = W_full[1:]

        print(f"   ✅ Closed-form solution — {len(self.weights)} weights")
        if feature_names is not None:
            top3 = np.argsort(np.abs(self.weights))[::-1][:3]
            for rank, idx in enumerate(top3, 1):
                print(f"      Top-{rank} weight: {feature_names[idx]} = {self.weights[idx]:.4f}")

    # ------------------------------------------------------------------
    # readable summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"Multiple Linear Regression (from scratch)\n"
            f"  {len(self.weights)} features\n"
            f"  bias = {self.bias:.4f}\n"
            f"  weight range = [{self.weights.min():.4f}, {self.weights.max():.4f}]\n"
            f"  Trained for {len(self.history.get('loss', []))} epochs"
        )
