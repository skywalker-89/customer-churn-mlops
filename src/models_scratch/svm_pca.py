import numpy as np
from .base import BaseModel
from .pca import PCAScratch
from .svm import SVMScratch

class SVMPCAScratch(BaseModel):
    def __init__(self, n_components=10, C=1.0, kernel="linear",
                 batch_size=2048, class_weight="balanced",
                 lr_decay=True, random_state=42, verbose=0):
        super().__init__()
        self.n_components = int(n_components)

        self.pca = PCAScratch(n_components=self.n_components)
        self.svm = SVMScratch(
            C=C, kernel=kernel,
            standardize=False,          # already standardized before PCA
            batch_size=batch_size,
            class_weight=class_weight,
            lr_decay=lr_decay,
            random_state=random_state,
            verbose=verbose
        )

    def fit(self, X, y, epochs=100, lr=0.001, feature_names=None, warm_start=False, **kwargs):
        self.feature_names = feature_names

        # standardize before PCA
        if warm_start and (self.nan_mean is not None) and (self.mu is not None) and (self.std is not None):
            Xp = self._prep_transform(X, standardize=True)
        else:
            Xp = self._prep_fit(X, standardize=True)

        pca_ready = getattr(self.pca, "components_", None) is not None
        if (not warm_start) or (not pca_ready):
            Z = self.pca.fit_transform(Xp)
        else:
            Z = self.pca.transform(Xp)

        self.svm.fit(Z, y, epochs=epochs, lr=lr, feature_names=feature_names, warm_start=warm_start)

    def predict(self, X):
        Xp = self._prep_transform(X, standardize=True)
        Z = self.pca.transform(Xp)
        return self.svm.predict(Z)
