import numpy as np
from .base import BaseModel

class CustomModelScratch(BaseModel):
    """
    AdaBoost (scratch) with decision stumps + class-imbalance aware weights
    + threshold tuning for best F1.
    epochs is treated as number of boosting rounds (T).
    """
    def __init__(self, n_estimators=80, learning_rate=0.5, val_size=0.15, random_state=42, n_thresholds=9):
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.val_size = float(val_size)
        self.random_state = int(random_state)
        self.n_thresholds = int(n_thresholds)

        self.stumps = []
        self.alphas = []
        self.threshold = 0.5

    def _rng(self):
        return np.random.RandomState(self.random_state)

    def _train_val_split(self, X, y, rng):
        n = len(y)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(np.floor(n * self.val_size))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    def _sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def _f1(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    def _predict_stump(self, X, feature, thr, polarity):
        x = X[:, feature]
        pred = np.ones(X.shape[0], dtype=np.int32)
        pred[x < thr] = -1
        if polarity == -1:
            pred *= -1
        return pred

    def _best_stump(self, X, y_sign, w):
        n, d = X.shape
        best = None
        best_err = np.inf

        qs = np.linspace(0.1, 0.9, self.n_thresholds)

        for j in range(d):
            col = X[:, j]
            thrs = np.unique(np.quantile(col, qs))
            for thr in thrs:
                pred = self._predict_stump(X, j, thr, polarity=+1)
                err = np.sum(w[pred != y_sign])
                if err < best_err:
                    best_err = err
                    best = (j, float(thr), +1)

                pred = -pred
                err = np.sum(w[pred != y_sign])
                if err < best_err:
                    best_err = err
                    best = (j, float(thr), -1)

        return best, float(best_err)

    def fit(self, X, y, epochs=100, lr=0.01, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names
        rng = self._rng()

        # for stumps: no standardize, but do NaN impute
        Xp = self._prep_fit(X, standardize=False) if (not warm_start or self.nan_mean is None) \
             else self._prep_transform(X, standardize=False)

        y = np.asarray(y, dtype=np.int32)

        # epochs = boosting rounds
        T = int(epochs) if (epochs and epochs > 0) else self.n_estimators
        eta = float(lr) if (lr and lr > 0) else self.learning_rate

        X_tr, y_tr, X_val, y_val = self._train_val_split(Xp, y, rng)

        if not warm_start:
            self.stumps = []
            self.alphas = []

        y_sign = np.where(y_tr == 1, 1, -1).astype(np.int32)

        # class-balanced initial weights
        idx_pos = np.where(y_tr == 1)[0]
        idx_neg = np.where(y_tr == 0)[0]
        w = np.zeros(len(y_tr), dtype=np.float64)
        if len(idx_pos) > 0:
            w[idx_pos] = 0.5 / len(idx_pos)
        if len(idx_neg) > 0:
            w[idx_neg] = 0.5 / len(idx_neg)
        if w.sum() == 0:
            w[:] = 1.0 / len(w)

        start_t = len(self.stumps)
        for _ in range(start_t, T):
            stump, err = self._best_stump(X_tr, y_sign, w)
            if stump is None:
                break
            err = max(1e-12, min(err, 1.0 - 1e-12))
            alpha = 0.5 * np.log((1.0 - err) / err) * eta

            j, thr, pol = stump
            pred = self._predict_stump(X_tr, j, thr, pol)

            w *= np.exp(-alpha * y_sign * pred)
            w /= (w.sum() + 1e-12)

            self.stumps.append({"feature": j, "thr": thr, "polarity": pol})
            self.alphas.append(float(alpha))

        # threshold tuning on validation
        proba_val = self.predict_proba(X_val)
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            pred = (proba_val >= t).astype(np.int32)
            f1 = self._f1(y_val, pred)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if len(self.stumps) == 0:
            return np.zeros(X.shape[0], dtype=np.float64)
        score = np.zeros(X.shape[0], dtype=np.float64)
        for stump, a in zip(self.stumps, self.alphas):
            pred = self._predict_stump(X, stump["feature"], stump["thr"], stump["polarity"]).astype(np.float64)
            score += a * pred
        return score

    def predict_proba(self, X):
        Xp = self._prep_transform(X, standardize=False)
        score = self.decision_function(Xp)
        return self._sigmoid(2.0 * score)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(np.int32)
