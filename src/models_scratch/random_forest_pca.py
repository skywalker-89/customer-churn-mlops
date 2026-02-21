import numpy as np
from .base import BaseModel
from .pca import PCAScratch
from .random_forest import RandomForestScratch

class RandomForestPCAScratch(BaseModel):
    def __init__(self, n_components=10, n_estimators=50, max_depth=10, min_samples_split=20, random_state=42):
        super().__init__()
        self.n_components = int(n_components)
        self.random_state = int(random_state)

        self.pca = PCAScratch(n_components=self.n_components)
        self.rf = RandomForestScratch(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

    def fit(self, X, y, epochs=0, lr=0.0, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names

        if warm_start and (self.nan_mean is not None):
            Xp = self._prep_transform(X, standardize=True)
        else:
            Xp = self._prep_fit(X, standardize=True)

        # PCA fit only if not warm_start
        if (not warm_start) or (self.pca.components_ is None):
            Z = self.pca.fit_transform(Xp)
        else:
            Z = self.pca.transform(Xp)

        self.rf.fit(Z, y, feature_names=feature_names, warm_start=warm_start)

    def predict(self, X):
        Xp = self._prep_transform(X, standardize=True)
        Z = self.pca.transform(Xp)
        return self.rf.predict(Z)
