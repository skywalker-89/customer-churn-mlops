import numpy as np
from .base import BaseModel
from .decision_tree import DecisionTreeScratch


class RandomForestScratch(BaseModel):
    def __init__(
        self,
        n_estimators=50,
        max_depth=10,
        min_samples_split=20,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    ):
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)

        self.trees = []

    def fit(
        self, X, y, epochs=0, lr=0.0, feature_names=None, warm_start=False, **kwargs
    ):
        # epochs/lr ignored (forest builds trees)
        self.feature_names = feature_names

        Xp = (
            self._prep_fit(X, standardize=False)
            if (not warm_start or self.nan_mean is None)
            else self._prep_transform(X, standardize=False)
        )

        y = np.asarray(y, dtype=np.int32).reshape(-1)

        rng = np.random.RandomState(self.random_state)
        n = Xp.shape[0]

        if (not warm_start) or (len(self.trees) == 0):
            self.trees = []

        start = len(self.trees)
        for t in range(start, self.n_estimators):
            if self.bootstrap:
                idx = rng.randint(0, n, size=n)
            else:
                idx = rng.choice(n, size=n, replace=False)

            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=rng.randint(0, 10**9),
            )
            tree.fit(Xp[idx], y[idx], feature_names=feature_names, warm_start=False)
            self.trees.append(tree)

        # Calculate feature importances
        if len(self.trees) > 0:
            self.feature_importances_ = np.zeros(Xp.shape[1])
            for tree in self.trees:
                if hasattr(tree, "feature_importances_"):
                    self.feature_importances_ += tree.feature_importances_
            self.feature_importances_ /= len(self.trees)

    def predict(self, X):
        Xp = self._prep_transform(X, standardize=False)
        # majority vote
        votes = np.zeros((Xp.shape[0], len(self.trees)), dtype=np.int32)
        for i, tree in enumerate(self.trees):
            votes[:, i] = tree.predict(Xp)
        return (np.mean(votes, axis=1) >= 0.5).astype(np.int32)

    def predict_proba(self, X):
        Xp = self._prep_transform(X, standardize=False)
        votes = np.zeros((Xp.shape[0], len(self.trees)), dtype=np.int32)
        for i, tree in enumerate(self.trees):
            # Assume tree.predict returns class labels (0 or 1)
            votes[:, i] = tree.predict(Xp)

        # Fraction of trees voting for class 1
        proba_1 = np.mean(votes, axis=1)
        # Return [prob_0, prob_1] for each sample
        return np.column_stack([1 - proba_1, proba_1])
