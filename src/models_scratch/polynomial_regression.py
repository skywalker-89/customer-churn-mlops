"""
Polynomial Regression — From Scratch (NumPy only)

Expands features to polynomial degree, then fits y = X_poly @ W + b
using gradient descent on MSE loss.

Course requirement: "functions built from scratch"
"""

import numpy as np
from itertools import combinations_with_replacement
from src.models_scratch.base import BaseModel


class PolynomialRegressionScratch(BaseModel):
    """
    Polynomial Regression using gradient descent.

    Steps:
      1. Expand X into polynomial features (degree 2/3/etc.)
      2. Run gradient descent on the expanded matrix

    Supports interaction terms and higher-degree terms.
    """

    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree
        # Scaling params
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None
        # Store poly feature metadata for predict()
        self._orig_n_features = None
        self._poly_feature_names = None

    # ------------------------------------------------------------------
    # polynomial feature expansion (from scratch)
    # ------------------------------------------------------------------
    def _expand_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate polynomial features up to self.degree.

        For degree=2 with features [x1, x2]:
          → [x1, x2, x1², x1*x2, x2²]

        For degree=3 adds cubic terms too.
        """
        n, d = X.shape
        features = [X]  # Start with original features
        names = [f"x{j}" for j in range(d)]

        for deg in range(2, self.degree + 1):
            # All combinations of deg features (with replacement)
            for combo in combinations_with_replacement(range(d), deg):
                col = np.ones(n, dtype=np.float32)
                name_parts = []
                for idx in combo:
                    col *= X[:, idx]
                    name_parts.append(f"x{idx}")
                features.append(col.reshape(-1, 1))
                names.append("*".join(name_parts))

        self._poly_feature_names = names
        return np.hstack(features).astype(np.float32)

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.00001,
        batch_size: int = 4096,
        feature_names: list = None,
        warm_start: bool = False,
    ) -> None:
        """
        Train polynomial regression via Mini-batch Gradient Descent.

        Parameters
        ----------
        X : (n_samples, n_features) original features
        y : (n_samples,) target
        epochs : number of epochs (passes over the full dataset)
        lr : learning rate
        batch_size : size of mini-batches
        warm_start : if True and weights exist, continue from previous training
        feature_names : optional column names
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, self._orig_n_features = X.shape

        # 1. Compute scaling statistics on a sample (to avoid OOM on full expansion)
        sample_size = min(50000, n_samples)
        print(f"   📊 Computing scaler stats on {sample_size} random samples...")
        indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[indices]
        X_sample_poly = self._expand_features(X_sample)
        
        self._X_mean = X_sample_poly.mean(axis=0)
        self._X_std = X_sample_poly.std(axis=0)
        self._X_std[self._X_std == 0] = 1.0
        
        d = X_sample_poly.shape[1]
        print(f"   Expanded {self._orig_n_features} features → {d} polynomial features (degree={self.degree})")

        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1.0

        # 2. Initialise parameters
        if warm_start and self.weights is not None and self.bias is not None:
             if len(self.weights) == d:
                # Convert back to normalized space for training
                W = self.weights * self._X_std / self._y_std
                b = (self.bias - self._y_mean) / self._y_std + np.sum(W * self._X_mean / self._X_std)
                print(f"   ♻️  Warm start: continuing from existing {d} weights")
             else:
                W = np.zeros(d, dtype=np.float32)
                b = 0.0
                print(f"   ⚠️  Dimension mismatch! Cold start.")
        else:
            W = np.zeros(d, dtype=np.float32)
            b = 0.0
            print(f"   🆕 Cold start: initializing {d} weights to zero")

        self.history["loss"] = []
        n_batches = int(np.ceil(n_samples / batch_size))

        # 3. Mini-batch Gradient Descent
        print(f"   🚀 Starting Mini-batch SGD: {epochs} epochs, {n_batches} batches/epoch")
        
        for epoch in range(epochs):
            # Shuffle data at start of epoch
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            epoch_loss = 0.0
            
            for i in range(0, n_samples, batch_size):
                X_batch_raw = X_shuffled[i:i+batch_size]
                y_batch_raw = y_shuffled[i:i+batch_size]
                
                # Expand and normalize ONLY the batch
                X_batch_poly = self._expand_features(X_batch_raw)
                X_batch = (X_batch_poly - self._X_mean) / self._X_std
                y_batch = (y_batch_raw - self._y_mean) / self._y_std
                
                # Forward pass
                y_hat = X_batch @ W + b
                error = y_hat - y_batch
                
                loss = float(np.mean(error ** 2))
                epoch_loss += loss * len(y_batch) # Weighted sum
                
                # Backward pass
                batch_n = len(y_batch)
                dW = (2.0 / batch_n) * (X_batch.T @ error)
                db = (2.0 / batch_n) * np.sum(error)
                
                W -= lr * dW
                b -= lr * db
            
            avg_epoch_loss = epoch_loss / n_samples
            self.history["loss"].append(avg_epoch_loss)
            
            if np.isnan(avg_epoch_loss) or np.isinf(avg_epoch_loss):
                print(f"      ⚠️  Divergence at epoch {epoch}! Try smaller lr.")
                break
                
            if epoch % max(1, epochs // 5) == 0:
                print(f"      Epoch {epoch:>3d}/{epochs}  MSE(norm)={avg_epoch_loss:.6f}")

        # 4. Convert normalised weights → original scale
        self.weights = self._y_std * W / self._X_std
        self.bias = float(
            self._y_std * (b - np.sum(W * self._X_mean / self._X_std)) + self._y_mean
        )

        print(f"   ✅ Training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict: expand to poly features, then W @ X_poly + b."""
        X = np.asarray(X, dtype=np.float32)
        # Process in chunks to avoid OOM during prediction if X is large
        n_samples = len(X)
        chunk_size = 4096
        y_pred = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(0, n_samples, chunk_size):
            X_chunk = X[i:i+chunk_size]
            X_poly_chunk = self._expand_features(X_chunk)
            y_pred[i:i+chunk_size] = X_poly_chunk @ self.weights + self.bias
            
        return y_pred

    # ------------------------------------------------------------------
    # readable summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        n_poly = len(self.weights) if self.weights is not None else 0
        return (
            f"Polynomial Regression (from scratch, degree={self.degree})\n"
            f"  {self._orig_n_features} original → {n_poly} polynomial features\n"
            f"  bias = {self.bias:.4f}\n"
            f"  Trained for {len(self.history.get('loss', []))} epochs"
        )
