import numpy as np
from .base import BaseModel

class AgglomerativeClusteringScratch(BaseModel):
    """
    Approx hierarchical clustering:
    - merge on a sample (greedy nearest centroid)
    - refine centroids on full train for 'epochs' steps
    - binary: tune margin threshold for better F1
    """
    def __init__(self, n_clusters=2, linkage="ward", sample_size=1500,
                 refine_iters=10, tol=1e-4, random_state=42):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.linkage = str(linkage)
        self.sample_size = int(sample_size)
        self.refine_iters = int(refine_iters)
        self.tol = float(tol)
        self.random_state = int(random_state)

        self.centroids = None

        self.pos_cluster = None
        self.neg_cluster = None
        self.margin_threshold = 0.0

    def _dist2(self, X, C):
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    def _assign(self, X, C):
        return np.argmin(self._dist2(X, C), axis=1)

    def _f1(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    def fit(self, X, y, epochs=100, lr=0.0, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names
        rng = np.random.RandomState(self.random_state)

        Xs = self._prep_fit(X, standardize=True) if (not warm_start or self.nan_mean is None) \
             else self._prep_transform(X, standardize=True)
        y = np.asarray(y, dtype=np.int32).reshape(-1)

        refine_steps = int(epochs) if (epochs and epochs > 0) else self.refine_iters

        # warm_start: reuse centroids and only refine
        if warm_start and (self.centroids is not None) and (self.centroids.shape[0] == self.n_clusters):
            C = self.centroids.copy()
        else:
            n = Xs.shape[0]
            m = min(self.sample_size, n)
            idx = rng.choice(n, size=m, replace=False)
            S = Xs[idx]

            # start each point as its own centroid (sample)
            C = S.copy()
            sizes = np.ones(m, dtype=np.int32)

            # greedy merge until k
            while C.shape[0] > self.n_clusters:
                D = self._dist2(C, C)
                np.fill_diagonal(D, np.inf)
                i, j = np.unravel_index(np.argmin(D), D.shape)
                if j < i:
                    i, j = j, i

                new_size = sizes[i] + sizes[j]
                new_c = (C[i] * sizes[i] + C[j] * sizes[j]) / new_size

                C[i] = new_c
                sizes[i] = new_size
                C = np.delete(C, j, axis=0)
                sizes = np.delete(sizes, j, axis=0)

        # refine on full train
        for _ in range(refine_steps):
            labels = self._assign(Xs, C)
            new_C = C.copy()
            for k in range(self.n_clusters):
                pts = Xs[labels == k]
                new_C[k] = pts.mean(axis=0) if len(pts) else Xs[rng.randint(0, Xs.shape[0])]
            shift = float(np.sqrt(((new_C - C) ** 2).sum()))
            C = new_C
            if shift < self.tol:
                break

        self.centroids = C

        # map clusters -> label + margin threshold
        cl = self._assign(Xs, self.centroids)
        rate = []
        for k in range(self.n_clusters):
            ys = y[cl == k]
            rate.append(float(ys.mean()) if len(ys) else 0.0)

        if self.n_clusters == 2:
            self.pos_cluster = int(np.argmax(rate))
            self.neg_cluster = 1 - self.pos_cluster

            D = self._dist2(Xs, self.centroids)
            score = D[:, self.neg_cluster] - D[:, self.pos_cluster]

            best_t, best_f1 = 0.0, -1.0
            candidates = np.quantile(score, np.linspace(0.05, 0.95, 19))
            for t in candidates:
                pred = (score >= t).astype(np.int32)
                f1 = self._f1(y, pred)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            self.margin_threshold = best_t
        else:
            self.pos_cluster = None
            self.neg_cluster = None
            self.margin_threshold = 0.0

    def predict(self, X):
        Xs = self._prep_transform(X, standardize=True)

        if self.n_clusters == 2 and (self.pos_cluster is not None):
            D = self._dist2(Xs, self.centroids)
            score = D[:, self.neg_cluster] - D[:, self.pos_cluster]
            return (score >= self.margin_threshold).astype(np.int32)

        cl = self._assign(Xs, self.centroids)
        return (cl == 0).astype(np.int32)
