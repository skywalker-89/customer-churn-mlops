"""
Ridge Regression — From Scratch (NumPy only)

Multiple linear regression with L2 regularisation:
  Loss = MSE + alpha * ||W||²

Provides BOTH:
  - Gradient descent solution
  - Closed-form solution: W = (X^T X + alpha*I)^{-1} X^T y

This is the "new model" outside classroom learning.

Course requirement: "new model built from scratch"
"""

import numpy as np
from src.models_scratch.base import BaseModel


class RidgeRegressionScratch(BaseModel):
    """
    Ridge Regression (L2 Regularised Linear Regression) from scratch.

    The L2 penalty shrinks weights toward zero, reducing overfitting and
    improving generalisation — especially when features are correlated.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        # Scaling params
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

    # ------------------------------------------------------------------
    # core API — gradient descent
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
        Train Ridge regression via gradient descent.

        Loss = (1/n) * ||y - Xw - b||² + alpha * ||w||²

        Gradient with respect to W:
          dW = (2/n) * X^T @ (y_hat - y) + 2*alpha*W

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        epochs : number of iterations
        lr : learning rate
        warm_start : if True and weights exist, continue from previous training
        feature_names : optional column names
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape

        # 1. Z-score normalisation
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std == 0] = 1.0

        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1.0

        X_norm = (X - self._X_mean) / self._X_std
        y_norm = (y - self._y_mean) / self._y_std

        # Scale alpha to normalised space
        alpha_scaled = self.alpha / self._y_std

        # 2. Initialise (with warm-start support)
        if warm_start and self.weights is not None and self.bias is not None:
            # Check dimension compatibility (features may have changed)
            if len(self.weights) == d:
                # Continue from existing weights (convert back to normalized space)
                W = self.weights * self._X_std / self._y_std
                b = (self.bias - self._y_mean) / self._y_std + np.sum(W * self._X_mean / self._X_std)
                print(f"   ♻️  Warm start: continuing from existing Ridge weights (alpha={self.alpha})")
            else:
                # Dimension mismatch - fall back to cold start
                W = np.zeros(d)
                b = 0.0
                print(f"   ⚠️  Dimension mismatch! Old model has {len(self.weights)} weights, need {d}. Cold start.")
        else:
            # Start from scratch
            W = np.zeros(d)
            b = 0.0
            print(f"   🆕 Cold start: initializing {d} Ridge weights to zero (alpha={self.alpha})")

        self.history["loss"] = []
        self.history["penalty"] = []

        # 3. Gradient descent with L2 penalty
        for epoch in range(epochs):
            y_hat = X_norm @ W + b
            error = y_hat - y_norm

            mse_loss = float(np.mean(error ** 2))
            l2_penalty = float(alpha_scaled * np.sum(W ** 2))
            total_loss = mse_loss + l2_penalty

            self.history["loss"].append(total_loss)
            self.history["penalty"].append(l2_penalty)

            # Gradients  (the L2 term adds 2*alpha*W to the weight gradient)
            dW = (2.0 / n) * (X_norm.T @ error) + 2.0 * alpha_scaled * W
            db = (2.0 / n) * np.sum(error)  # bias is NOT regularised

            W -= lr * dW
            b -= lr * db

            if epoch % max(1, epochs // 5) == 0:
                print(
                    f"      Epoch {epoch:>5d}/{epochs}  "
                    f"loss={total_loss:.6f}  (MSE={mse_loss:.6f} + L2={l2_penalty:.6f})"
                )

        # 4. Convert to original scale
        self.weights = self._y_std * W / self._X_std
        self.bias = float(
            self._y_std * (b - np.sum(W * self._X_mean / self._X_std)) + self._y_mean
        )

        print(f"   ✅ Training complete (alpha={self.alpha}) — {d} weights")
        if feature_names is not None:
            top3 = np.argsort(np.abs(self.weights))[::-1][:3]
            for rank, idx in enumerate(top3, 1):
                print(f"      Top-{rank}: {feature_names[idx]} = {self.weights[idx]:.4f}")

    # ------------------------------------------------------------------
    # closed-form solution
    # ------------------------------------------------------------------
    def fit_closed_form(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None,
    ) -> None:
        """
        Solve Ridge analytically:
          W = (X^T X + alpha * I)^{-1} X^T y    (no bias column in I)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Add bias column
        X_b = self._add_bias(X)  # (n, d+1)  — col-0 is bias
        d_plus_1 = X_b.shape[1]

        # Identity but don't penalise the bias
        I = np.eye(d_plus_1)
        I[0, 0] = 0.0  # no regularisation on bias

        W_full = np.linalg.solve(X_b.T @ X_b + self.alpha * I, X_b.T @ y)

        self.bias = float(W_full[0])
        self.weights = W_full[1:]

        print(f"   ✅ Closed-form Ridge (alpha={self.alpha}) — {len(self.weights)} weights")
        if feature_names is not None:
            top3 = np.argsort(np.abs(self.weights))[::-1][:3]
            for rank, idx in enumerate(top3, 1):
                print(f"      Top-{rank}: {feature_names[idx]} = {self.weights[idx]:.4f}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using learned weights and bias."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias

    # ------------------------------------------------------------------
    # readable summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"Ridge Regression (from scratch, alpha={self.alpha})\n"
            f"  {len(self.weights)} features\n"
            f"  bias = {self.bias:.4f}\n"
            f"  L2 weight norm = {np.sum(self.weights**2):.6f}\n"
            f"  Trained for {len(self.history.get('loss', []))} epochs"
        )
