import numpy as np
from .base import BaseModel

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, feature=-1, threshold=0.0, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeScratch(BaseModel):
    def __init__(self, max_depth=10, min_samples_split=20, n_thresholds=16, max_features=None, random_state=42):
        super().__init__()
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.n_thresholds = int(n_thresholds)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.root = None

    @staticmethod
    def _gini(y):
        if y.size == 0:
            return 0.0
        p1 = np.mean(y == 1)
        p0 = 1.0 - p1
        return 1.0 - (p0 * p0 + p1 * p1)

    @staticmethod
    def _majority(y):
        return int(np.sum(y == 1) >= np.sum(y == 0))

    def _pick_features(self, d, rng):
        if self.max_features is None:
            return np.arange(d)

        if isinstance(self.max_features, int):
            k = max(1, min(d, self.max_features))
        elif isinstance(self.max_features, str) and self.max_features == "sqrt":
            k = max(1, int(np.sqrt(d)))
        elif isinstance(self.max_features, str) and self.max_features == "log2":
            k = max(1, int(np.log2(d)))
        else:
            k = d
        return rng.choice(d, size=k, replace=False)

    def _best_split(self, X, y, feat_idx):
        best_f, best_thr, best_score = -1, None, 1e18
        parent = self._gini(y)
        if parent == 0.0:
            return -1, None

        n = X.shape[0]
        qs = np.linspace(0.05, 0.95, self.n_thresholds)

        for f in feat_idx:
            col = X[:, f]
            thrs = np.unique(np.quantile(col, qs))
            for thr in thrs:
                left = col <= thr
                right = ~left
                if left.sum() < 1 or right.sum() < 1:
                    continue
                score = (left.sum() / n) * self._gini(y[left]) + (right.sum() / n) * self._gini(y[right])
                if score < best_score:
                    best_score, best_f, best_thr = score, f, float(thr)

        if best_f == -1 or best_score >= parent:
            return -1, None
        return best_f, best_thr

    def _build(self, X, y, depth, rng):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split or self._gini(y) == 0.0:
            return _Node(value=self._majority(y))

        feat_idx = self._pick_features(X.shape[1], rng)
        f, thr = self._best_split(X, y, feat_idx)
        if f == -1:
            return _Node(value=self._majority(y))

        left = X[:, f] <= thr
        right = ~left

        node = _Node(feature=f, threshold=thr)
        node.left = self._build(X[left], y[left], depth + 1, rng)
        node.right = self._build(X[right], y[right], depth + 1, rng)
        return node

    def fit(self, X, y, epochs=0, lr=0.0, feature_names=None, warm_start=False, **kwargs):
        # epochs/lr ignored (tree builds once)
        self.feature_names = feature_names
        rng = np.random.RandomState(self.random_state)

        Xp = self._prep_fit(X, standardize=False) if (not warm_start or self.nan_mean is None) \
             else self._prep_transform(X, standardize=False)

        y = np.asarray(y, dtype=np.int32).reshape(-1)
        self.root = self._build(Xp, y, 0, rng)

    def _pred_one(self, x):
        node = self.root
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        Xp = self._prep_transform(X, standardize=False)
        return np.array([self._pred_one(x) for x in Xp], dtype=np.int32)
