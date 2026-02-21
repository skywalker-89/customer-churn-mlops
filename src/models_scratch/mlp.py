import numpy as np
from .base import BaseModel

class MLPScratch(BaseModel):
    """
    MLP (scratch) for binary classification:
    - standardize (BaseModel)
    - mini-batch GD
    - class imbalance weighting
    - threshold tuning (F1) on validation
    """
    def __init__(self, hidden_layers=(64, 32), activation="relu",
                 batch_size=1024, val_size=0.15, random_state=42):
        super().__init__()
        self.hidden_layers = list(hidden_layers)
        self.activation = str(activation)
        self.batch_size = int(batch_size)
        self.val_size = float(val_size)
        self.random_state = int(random_state)

        self.W = []
        self.b = []
        self.threshold = 0.5

    def _rng(self):
        return np.random.RandomState(self.random_state)

    def _act(self, z):
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        return np.maximum(0.0, z)  # relu

    def _act_grad(self, z):
        if self.activation == "tanh":
            a = np.tanh(z)
            return 1.0 - a * a
        if self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        return (z > 0).astype(np.float64)  # relu

    def _sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def _init_params(self, n_in):
        rng = self._rng()
        self.W = []
        self.b = []
        sizes = [n_in] + self.hidden_layers + [1]
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
            self.W.append(rng.randn(sizes[i], sizes[i + 1]) * scale)
            self.b.append(np.zeros((1, sizes[i + 1]), dtype=np.float64))

    def _train_val_split(self, X, y):
        rng = self._rng()
        n = len(y)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(np.floor(n * self.val_size))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    def _f1(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    def fit(self, X, y, epochs=100, lr=0.001, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names
        rng = self._rng()

        # warm_start: don't refit scaler
        Xs = self._prep_fit(X, standardize=True) if (not warm_start or self.nan_mean is None) \
             else self._prep_transform(X, standardize=True)

        y = np.asarray(y, dtype=np.int32).reshape(-1, 1)

        X_tr, y_tr, X_val, y_val = self._train_val_split(Xs, y)

        if (not warm_start) or (len(self.W) == 0):
            self._init_params(X_tr.shape[1])

        pos = float(np.sum(y_tr == 1))
        neg = float(np.sum(y_tr == 0))
        pos_weight = (neg / pos) if pos > 0 else 1.0

        self.history = {"loss": []}

        n = X_tr.shape[0]
        bs = self.batch_size

        for _ in range(int(epochs)):
            idx = np.arange(n)
            rng.shuffle(idx)

            total_loss = 0.0
            for start in range(0, n, bs):
                bidx = idx[start:start + bs]
                xb = X_tr[bidx]
                yb = y_tr[bidx]

                # forward
                A = [xb]
                Z = []
                for i in range(len(self.hidden_layers)):
                    z = A[-1] @ self.W[i] + self.b[i]
                    Z.append(z)
                    A.append(self._act(z))

                z_out = A[-1] @ self.W[-1] + self.b[-1]
                p = self._sigmoid(z_out)

                w = np.where(yb == 1, pos_weight, 1.0)
                loss = -np.mean(w * (yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12)))
                total_loss += loss * len(bidx)

                # backward
                dZout = (p - yb) * w / len(bidx)

                dW = [None] * len(self.W)
                db = [None] * len(self.b)

                dW[-1] = A[-1].T @ dZout
                db[-1] = np.sum(dZout, axis=0, keepdims=True)

                dA = dZout @ self.W[-1].T

                for i in reversed(range(len(self.hidden_layers))):
                    dz = dA * self._act_grad(Z[i])
                    dW[i] = A[i].T @ dz
                    db[i] = np.sum(dz, axis=0, keepdims=True)
                    if i > 0:
                        dA = dz @ self.W[i].T

                for i in range(len(self.W)):
                    self.W[i] -= lr * dW[i]
                    self.b[i] -= lr * db[i]

            self.history["loss"].append(total_loss / n)

        # threshold tuning on validation
        proba_val = self.predict_proba_raw(X_val)
        best_t, best_f1 = 0.5, -1.0
        yv = y_val.reshape(-1)
        for t in np.linspace(0.05, 0.95, 19):
            pred = (proba_val >= t).astype(np.int32)
            f1 = self._f1(yv, pred)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t

    def predict_proba_raw(self, Xs):
        A = Xs
        for i in range(len(self.hidden_layers)):
            A = self._act(A @ self.W[i] + self.b[i])
        z_out = A @ self.W[-1] + self.b[-1]
        return self._sigmoid(z_out).reshape(-1)

    def predict(self, X):
        Xs = self._prep_transform(X, standardize=True)
        proba = self.predict_proba_raw(Xs)
        return (proba >= self.threshold).astype(np.int32)
