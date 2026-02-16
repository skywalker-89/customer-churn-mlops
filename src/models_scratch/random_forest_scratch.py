import numpy as np
from src.models_scratch.base import BaseModel
from src.models_scratch.xgboost import DecisionTreeScratch

class RandomForestScratch(BaseModel):
    """
    Random Forest Regressor built from scratch.
    Uses Bootstrap Aggregation (Bagging) and Feature Randomness.
    """
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, max_features="sqrt"):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []
        # Required by BaseModel contract but not used
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = None, lr: float = None, 
            feature_names: list = None, warm_start: bool = False, **kwargs) -> None:
        """
        Train the Random Forest model.
        Args:
            epochs: Mapped to n_estimators for compatibility with benchmark script
        """
        # Map params if provided (overriding __init__)
        n_estimators = epochs if epochs is not None else self.n_estimators
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape
        
        if not warm_start or not self.trees:
            self.trees = []
            print(f"   🆕 Cold start: Initializing Random Forest")
        else:
            print(f"   ♻️  Warm start: Appending new trees to existing {len(self.trees)}")

        print(f"   🚀 Training RandomForest (Scratch): {n_estimators} trees, "
              f"depth={self.max_depth}, min_samples_leaf={self.min_samples_leaf}")

        # Optional: use max_samples to reduce tree size
        max_samples_ratio = kwargs.get('max_samples', 1.0)
        n_bootstrap = int(n_samples * max_samples_ratio)

        # Training Loop
        for i in range(n_estimators):
            # 1. Bootstrap Sampling (Bagging)
            # Randomly select n_samples with replacement
            idxs = np.random.choice(n_samples, n_bootstrap, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            
            # 2. Train Decision Tree
            # Note: DecisionTreeScratch already handles feature randomness internally if configured,
            # but we can also enforce it here if we modified DecisionTree. 
            # Our current DecisionTreeScratch implementation selects sqrt(n_features) at each split.
            tree = DecisionTreeScratch(max_depth=self.max_depth, 
                                     min_samples_split=self.min_samples_split,
                                     min_samples_leaf=self.min_samples_leaf,
                                     max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            
            # 3. Save tree
            self.trees.append(tree)
            
            if i % max(1, n_estimators // 5) == 0:
                print(f"      Tree {i:>3d}/{n_estimators} built")

        print(f"   ✅ Training complete! Total trees: {len(self.trees)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by averaging predictions from all trees."""
        X = np.asarray(X, dtype=np.float32)
        
        # Collect predictions from all trees
        # shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Average them
        return np.mean(tree_preds, axis=0)
