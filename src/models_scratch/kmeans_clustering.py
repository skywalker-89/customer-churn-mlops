import numpy as np
from .base import BaseModel


class KMeansScratch(BaseModel):
    """
    K-Means (scratch) used as a weak classifier:
    - Unsupervised clustering (k=2)
    - Uses y ONLY to map cluster->label (allowed in your benchmark)
    - Uses a calibrated margin threshold to avoid predicting almost all positives
    """

    def __init__(self, n_clusters=2, max_iters=100, tol=1e-4, n_init=5, random_state=42):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.random_state = int(random_state)

        self.centroids = None

        # mapping + threshold (binary)
        self.cluster_to_label = None
        self.pos_cluster = None
        self.neg_cluster = None
        self.margin_threshold = 0.0
        self.base_rate = None  # churn rate on train

    # ---------- helpers ----------
    def _dist2(self, X, C):
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    def _assign(self, X, C):
        return np.argmin(self._dist2(X, C), axis=1)

    def _inertia(self, X, C, labels):
        return float(((X - C[labels]) ** 2).sum())

    def _kmeanspp_init(self, X, rng):
        n = X.shape[0]
        C = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)

        C[0] = X[rng.randint(0, n)]
        d2 = ((X - C[0]) ** 2).sum(axis=1)

        for k in range(1, self.n_clusters):
            probs = d2 / (d2.sum() + 1e-12)
            idx = rng.choice(n, p=probs)
            C[k] = X[idx]
            new_d2 = ((X - C[k]) ** 2).sum(axis=1)
            d2 = np.minimum(d2, new_d2)

        return C

    # ---------- main ----------
    def fit(self, X, y, epochs=100, lr=0.0, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names
        rng = np.random.RandomState(self.random_state)

        # preprocess (impute + standardize) using BaseModel
        if warm_start and (self.nan_mean is not None) and (self.mu is not None) and (self.std is not None):
            Xs = self._prep_transform(X, standardize=True)
        else:
            Xs = self._prep_fit(X, standardize=True)

        y = np.asarray(y, dtype=np.int32).reshape(-1)
        self.base_rate = float(y.mean())  # training churn rate (e.g. 0.2369)

        iters = int(epochs) if (epochs and epochs > 0) else self.max_iters
        if iters <= 0:
            iters = self.max_iters

        best_C = None
        best_inertia = np.inf

        # warm_start centroids as one init (optional)
        init_list = []
        if warm_start and (self.centroids is not None) and (self.centroids.shape[0] == self.n_clusters):
            init_list.append(self.centroids.copy())

        # remaining inits by kmeans++
        for _ in range(max(0, self.n_init - len(init_list))):
            init_list.append(self._kmeanspp_init(Xs, rng))

        for C in init_list:
            for _t in range(iters):
                labels = self._assign(Xs, C)

                new_C = C.copy()
                for k in range(self.n_clusters):
                    pts = Xs[labels == k]
                    new_C[k] = pts.mean(axis=0) if pts.size else Xs[rng.randint(0, Xs.shape[0])]

                shift = float(np.sqrt(((new_C - C) ** 2).sum()))
                C = new_C
                if shift < self.tol:
                    break

            labels = self._assign(Xs, C)
            inertia = self._inertia(Xs, C, labels)
            if inertia < best_inertia:
                best_inertia = inertia
                best_C = C.copy()

        self.centroids = best_C
        self.history["best_inertia"] = float(best_inertia)

        # ---------- cluster -> label mapping ----------
        cl = self._assign(Xs, self.centroids)
        rate = np.zeros(self.n_clusters, dtype=np.float64)
        for k in range(self.n_clusters):
            ys = y[cl == k]
            rate[k] = float(ys.mean()) if ys.size else 0.0

        self.cluster_to_label = {k: int(rate[k] >= self.base_rate) for k in range(self.n_clusters)}

        # ---------- binary: calibrated margin threshold ----------
        self.pos_cluster = None
        self.neg_cluster = None
        self.margin_threshold = 0.0

        if self.n_clusters == 2:
            # which centroid corresponds to "more churn"
            self.pos_cluster = int(np.argmax(rate))
            self.neg_cluster = 1 - self.pos_cluster
            self.cluster_to_label = {self.pos_cluster: 1, self.neg_cluster: 0}

            # margin score: higher => closer to pos centroid
            D = self._dist2(Xs, self.centroids)
            score = D[:, self.neg_cluster] - D[:, self.pos_cluster]

            # CALIBRATION: choose threshold so predicted positive rate ~= base_rate
            # (prevents "predict almost everyone churn")
            q = 1.0 - self.base_rate
            q = float(np.clip(q, 0.05, 0.95))
            self.margin_threshold = float(np.quantile(score, q))

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("KMeansScratch not fitted yet (centroids is None).")

        Xs = self._prep_transform(X, standardize=True)

        # binary: use calibrated margin threshold
        if self.n_clusters == 2 and (self.pos_cluster is not None):
            D = self._dist2(Xs, self.centroids)
            score = D[:, self.neg_cluster] - D[:, self.pos_cluster]
            return (score >= self.margin_threshold).astype(np.int32)

        # general: use mapping
        cl = self._assign(Xs, self.centroids)
        map_arr = np.zeros(self.n_clusters, dtype=np.int32)
        for k, v in self.cluster_to_label.items():
            map_arr[int(k)] = int(v)
        return map_arr[cl].astype(np.int32)
