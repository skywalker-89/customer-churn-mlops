import numpy as np
from .base import BaseModel

class SVMScratch(BaseModel):
    """
    Linear soft-margin SVM (hinge loss) trained with mini-batch SGD.
    Fixes common "predict all zeros" issue by:
      - class_weight='balanced' (handles imbalance)
      - batch gradient (more stable bias than per-sample updates)
      - optional lr decay
    """
    def __init__(self, C=1.0, kernel="linear", standardize: bool = True,
                 batch_size: int = 2048, class_weight: str = "balanced",
                 lr_decay: bool = True, random_state: int = 42, verbose: int = 0):
        super().__init__()
        self.C = float(C)
        self.kernel = str(kernel)
        self.standardize = bool(standardize)

        self.batch_size = int(batch_size)
        self.class_weight = class_weight  # "balanced" or None
        self.lr_decay = bool(lr_decay)
        self.random_state = int(random_state)
        self.verbose = int(verbose)

        # stored for debugging
        self.C_pos = None
        self.C_neg = None

    def decision_function(self, X):
        Xp = self._prep_transform(X, standardize=self.standardize)
        return Xp @ self.weights + self.bias

    def fit(self, X, y, epochs=100, lr=0.001, feature_names=None, warm_start=False, **kwargs):
        if self.kernel != "linear":
            raise NotImplementedError("SVMScratch supports only kernel='linear'.")

        self.feature_names = feature_names

        # preprocess
        if warm_start and (self.nan_mean is not None):
            Xp = self._prep_transform(X, standardize=self.standardize)
        else:
            Xp = self._prep_fit(X, standardize=self.standardize)

        y = np.asarray(y, dtype=np.int32).reshape(-1)
        ys = np.where(y == 1, 1.0, -1.0).astype(np.float64)

        n, d = Xp.shape
        if (self.weights is None) or (not warm_start):
            self.weights = np.zeros(d, dtype=np.float64)
            self.bias = 0.0

        rng = np.random.RandomState(self.random_state)

        # ---- class weighting (prevents bias drifting to all-zeros) ----
        if self.class_weight == "balanced":
            pos = float(np.sum(y == 1))
            neg = float(np.sum(y == 0))
            # scale so positive updates matter more when positives are rarer
            self.C_pos = self.C * (neg / (pos + 1e-12))
            self.C_neg = self.C
            C_i = np.where(y == 1, self.C_pos, self.C_neg).astype(np.float64)
        else:
            self.C_pos = self.C_neg = self.C
            C_i = np.full(n, self.C, dtype=np.float64)

        bs = max(1, self.batch_size)

        for ep in range(int(epochs)):
            # optional lr decay
            lr_ep = float(lr) / np.sqrt(ep + 1.0) if self.lr_decay else float(lr)

            idx = np.arange(n)
            rng.shuffle(idx)

            for start in range(0, n, bs):
                bidx = idx[start:start + bs]
                Xb = Xp[bidx]
                yb = ys[bidx]
                Cb = C_i[bidx]

                # margins
                s = Xb @ self.weights + self.bias
                margin = yb * s
                mask = margin < 1.0

                # gradients of: 0.5||w||^2 + mean(C * hinge)
                grad_w = self.weights.copy()
                grad_b = 0.0

                if np.any(mask):
                    Xm = Xb[mask]
                    ym = yb[mask]
                    Cm = Cb[mask]

                    # -(C*y*x) averaged
                    grad_w -= (Xm.T @ (Cm * ym)) / len(bidx)
                    # d/db = -mean(C*y) -> update b += lr*mean(C*y)
                    grad_b = -np.sum(Cm * ym) / len(bidx)

                self.weights -= lr_ep * grad_w
                self.bias    -= lr_ep * grad_b

            if self.verbose:
                # quick sanity print every 10 epochs
                if (ep + 1) % 10 == 0:
                    pred_rate = float(np.mean((Xp @ self.weights + self.bias) >= 0.0))
                    print(f"   [SVM] epoch {ep+1}/{epochs} | lr={lr_ep:.5f} | pred_pos_rate={pred_rate:.3f}")

        self.history["trained_epochs"] = self.history.get("trained_epochs", 0) + int(epochs)

    def predict(self, X):
        s = self.decision_function(X)
        return (s >= 0.0).astype(np.int32)
