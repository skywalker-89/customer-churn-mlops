import numpy as np
from src.models_scratch.base import BaseModel

class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx  # Feature index to split on
        self.threshold = threshold      # Threshold value for split
        self.value = value              # Value if this is a leaf node
        self.left = left                # Left child
        self.right = right              # Right child

class DecisionTreeScratch:
    """A regression tree that fits to residuals."""
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            leaf_value = np.mean(y)
            return DecisionNode(value=leaf_value)

        # Find best split
        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is None:
            return DecisionNode(value=np.mean(y))

        # Split data
        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = ~left_idxs
        
        # Grow children
        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return DecisionNode(feature_idx=best_feat, threshold=best_thresh, 
                           left=left_child, right=right_child)

    def _best_split(self, X, y, n_features):
        best_mse = float("inf")
        best_feat, best_thresh = None, None
        
        # Determine number of features to consider
        if self.max_features == "sqrt":
            n_consider = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            n_consider = int(np.log2(n_features))
        elif self.max_features == "auto" or self.max_features is None:
            n_consider = n_features
        elif isinstance(self.max_features, int):
            n_consider = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            n_consider = int(self.max_features * n_features)
        else:
            n_consider = n_features

        # Loop through features (randomly select features)
        feat_idxs = np.random.choice(n_features, n_consider, replace=False)
        
        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            # Don't check every single threshold if there are too many (optimization)
            if len(thresholds) > 300:
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 100))
                
            for thresh in thresholds:
                left_idxs = X[:, feat_idx] < thresh
                
                # Check min_samples_leaf constraint
                if len(y[left_idxs]) < self.min_samples_leaf or len(y[~left_idxs]) < self.min_samples_leaf:
                    continue
                    
                mse = self._calculate_mse(y, left_idxs)
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat_idx
                    best_thresh = thresh
                if len(y[left_idxs]) == 0 or len(y[~left_idxs]) == 0:
                    continue
                    
                mse = self._calculate_mse(y, left_idxs)
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat_idx
                    best_thresh = thresh
                    
        return best_feat, best_thresh

    def _calculate_mse(self, y, left_idxs):
        y_left = y[left_idxs]
        y_right = y[~left_idxs]
        
        # Weighted MSE of children
        n = len(y)
        mse_left = np.var(y_left) * len(y_left)
        mse_right = np.var(y_right) * len(y_right)
        
        return (mse_left + mse_right) / n

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class XGBoostScratch(BaseModel):
    """
    Gradient Boosting Regressor built from scratch.
    Fits a sequence of Decision Trees to the residuals of previous trees.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
        self.weights = None # Not used, but required by BaseModel contract
        self.bias = None    # Not used
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = None, lr: float = None, 
            feature_names: list = None, warm_start: bool = False, **kwargs) -> None:
        """
        Train the Gradient Boosting model.
        Args:
            epochs: Mapped to n_estimators for compatibility
            lr: Mapped to learning_rate for compatibility
        """
        # Map params if provided (overriding __init__)
        n_estimators = epochs if epochs is not None else self.n_estimators
        learning_rate = lr if lr is not None else self.learning_rate
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        
        # 1. Z-score normalization for target (Critical for convergence)
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        if self._y_std == 0: self._y_std = 1.0
        
        y_norm = (y - self._y_mean) / self._y_std
        
        n_samples = len(y)
        self.history["loss"] = []
        
        # 0. Initial prediction (mean of NORMALIZED target => 0.0)
        self.initial_prediction = np.mean(y_norm) # Should be ~0.0
        
        if not warm_start or not self.trees:
            self.trees = []
            current_pred = np.full(n_samples, self.initial_prediction)
            print(f"   🆕 Cold start: Initializing with mean={self._y_mean:.4f} (norm={self.initial_prediction:.4f})")
        else:
            # Reconstruct prediction from existing trees
            current_pred = np.full(n_samples, self.initial_prediction)
            for tree in self.trees:
                current_pred += self.learning_rate * tree.predict(X)
            print(f"   ♻️  Warm start: Continuing with {len(self.trees)} trees")

        print(f"   🚀 Training XGBoost (Scratch): {n_estimators} estimators, lr={learning_rate}")

        # Gradient Boosting Loop
        for i in range(n_estimators):
            # 1. Calculate residuals (negative gradient for MSE)
            # Loss = 0.5 * (y - pred)^2  => Gradient = -(y - pred) => Residual = y - pred
            residuals = y_norm - current_pred
            
            # 2. Fit weak learner (Decision Tree) to residuals
            tree = DecisionTreeScratch(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # 3. Update predictions
            update = tree.predict(X)
            current_pred += learning_rate * update
            
            # 4. Save tree
            self.trees.append(tree)
            
            # Log loss (on normalized data)
            mse = np.mean((y_norm - current_pred) ** 2)
            self.history["loss"].append(mse)
            
            if i % max(1, n_estimators // 5) == 0:
                print(f"      Tree {i:>3d}/{n_estimators}  MSE(norm)={mse:.6f}")

        print(f"   ✅ Training complete! Total trees: {len(self.trees)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by summing up contributions from all trees."""
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        
        # Start with initial prediction (normalized space)
        final_pred_norm = np.full(n_samples, self.initial_prediction)
        
        # Add contributions from all trees
        for tree in self.trees:
            final_pred_norm += self.learning_rate * tree.predict(X)
            
        # Denormalize predictions
        return final_pred_norm * self._y_std + self._y_mean
