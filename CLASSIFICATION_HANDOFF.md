# 🎯 Classification Model Development - Handoff Instructions

## What You Need to Do

Build **11 classification models from scratch**, benchmark them against sklearn, and send me the trained models when they're ready.

---

## 📋 Models You Need to Build (From Scratch)

Build these **11 models** in the `src/models_scratch/` folder:

1. ✅ **Logistic Regression** (gradient descent)
2. ✅ **Decision Tree** (recursive splits)
3. ✅ **Random Forest** (ensemble of trees)
4. ✅ **Support Vector Machine (SVM)** (margin classifier)
5. ✅ **Random Forest + PCA** (dimensionality reduction)
6. ✅ **SVM + PCA** (dimensionality reduction)
7. ✅ **K-Means Clustering** (unsupervised, 2 clusters)
8. ✅ **Agglomerative Clustering** (unsupervised, hierarchical)
9. ✅ **Perceptron / Single-Layer Perceptron (SLP)**
10. ✅ **Multi-Layer Perceptron (MLP)** (neural network)
11. ✅ **Your Choice** (pick ANY model from outside class - e.g., Naive Bayes, AdaBoost, etc.)

**PLUS:**
12. 📦 **One sklearn model** for comparison (e.g., `RandomForestClassifier` or `XGBClassifier`)

---

## 🎯 What You're Predicting - Understanding Customer Churn

### **Target Variable: `churned`**
- **Type:** Binary classification (0 or 1)
- **Values:**
  - `0` = Customer **stayed** (retained, did NOT churn)
  - `1` = Customer **left** (churned)

### **What Does "Churn" Mean?**
**Customer churn** = When a customer **stops doing business** with the company.

**In retail context:**
- **Churned customer (1):** Stopped buying from the store, haven't purchased in a long time, likely won't come back
- **Retained customer (0):** Still actively shopping, loyal customer, continues to make purchases

**Why predict churn?**
- Identify customers at risk of leaving
- Take action to retain them (discounts, promotions, outreach)
- More cost-effective to keep existing customers than acquire new ones

### **Your Dataset - Same Features, Different Target**
✅ **You use the SAME `training_data.parquet` file that already exists!**

Your lead engineer (regression team) predicts: `total_sales` (how much money customers spend)  
You (classification team) predict: `churned` (whether customers will leave or stay)

**Same features (customer behavior, purchase history, demographics)**  
**Different prediction target!**

**NO new feature engineering needed - everything is ready!**

---

## 🔥 SUPER IMPORTANT - FAIRNESS REQUIREMENTS

> **🚨 CRITICAL:** For fairness, ALL models (your from-scratch AND sklearn) MUST:
> 
> 1. ✅ Use the **SAME train/test split** → *(automatically handled by benchmark)*
> 2. ✅ Use the **SAME features** → *(automatically handled by benchmark)*
> 3. ✅ **TRAIN FOR THE SAME NUMBER OF ITERATIONS (EPOCHS)** → ⚠️ **YOU MUST CONFIGURE THIS!**
>
> **This means:** If your Logistic Regression trains for 100 epochs, your sklearn model should also train with comparable iterations!

---

## 📝 Step-by-Step Instructions

### **STEP 1: Study How Models Work**

**Look at the base class first:**
```bash
cat src/models_scratch/base.py
```

Every model you build **MUST inherit from `BaseModel`** and implement:
- `fit(X, y, epochs, lr, feature_names, warm_start)` - Train the model
- `predict(X)` - Return predictions

**Study the regression examples:**
```bash
# Gradient descent example
cat src/models_scratch/linear_regression.py

# Neural network example  
cat src/models_scratch/polynomial_regression.py

# Tree-based example
cat src/models_scratch/xgboost.py
```

---

### **STEP 2: Build Each Model**

Create each model as a separate file in `src/models_scratch/`.

**Example: Logistic Regression**
```python
# File: src/models_scratch/logistic_regression.py

import numpy as np
from .base import BaseModel

class LogisticRegressionScratch(BaseModel):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, epochs=100, lr=0.01, feature_names=None, warm_start=False):
        n_samples, n_features = X.shape
        
        # Initialize weights
        if not warm_start or self.weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute loss
            loss = -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
            
            # Backward pass
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)
            
            # Update weights
            self.weights -= lr * dw
            self.bias -= lr * db
            
            if epoch % 10 == 0:
                print(f"      Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred > 0.5).astype(int)  # Binary: 0 or 1
```

**Repeat this for ALL 11 models!**

---

### **STEP 3: Configure the Benchmark File**

Open: `src/classification/retail_classification_benchmark.py`

**3A) Import your models** (around line 40):
```python
# 🔧 ADD YOUR IMPORTS HERE
from src.models_scratch.logistic_regression import LogisticRegressionScratch
from src.models_scratch.decision_tree import DecisionTreeScratch
from src.models_scratch.random_forest import RandomForestScratch
from src.models_scratch.svm import SVMScratch
from src.models_scratch.random_forest_pca import RandomForestPCAScratch
from src.models_scratch.svm_pca import SVMPCAScratch
from src.models_scratch.kmeans_clustering import KMeansScratch
from src.models_scratch.agglomerative_clustering import AgglomerativeClusteringScratch
from src.models_scratch.perceptron import PerceptronScratch
from src.models_scratch.mlp import MLPScratch
from src.models_scratch.your_custom_model import YourCustomModelScratch
```

**3B) Configure hyperparameters** (around line 60):
```python
# ⚙️ CONFIGURE HYPERPARAMETERS HERE
MODEL_CONFIG = {
    "LogisticRegression": {
        "epochs": 100,       # ← Number of training iterations
        "lr": 0.01,
        "warm_start_epochs": 20
    },
    "DecisionTree": {
        "max_depth": 10,
        "min_samples_split": 20,
        "warm_start_epochs": 0
    },
    # ... configure ALL your models here
}
```

**3C) Add model instances** (around line 310):
```python
# 🔧 ADD YOUR MODEL INSTANCES HERE
models = [
    ("LogisticRegression", 
     LogisticRegressionScratch(), 
     MODEL_CONFIG["LogisticRegression"]["epochs"], 
     MODEL_CONFIG["LogisticRegression"]["lr"]),
    
    ("DecisionTree", 
     DecisionTreeScratch(max_depth=MODEL_CONFIG["DecisionTree"]["max_depth"]), 
     0,  # Trees don't use epochs
     0),
    
    # ... add ALL your models here
]
```

**3D) Configure sklearn model** (around line 460):
```python
# 📦 CONFIGURE SKLEARN MODEL HERE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,    # ← MAKE SURE THIS MATCHES YOUR FROM-SCRATCH ITERATIONS!
    max_depth=10,
    random_state=42,
    n_jobs=3
)
model_name = "random_forest_sklearn"
```

> **🚨 REMEMBER:** Match the sklearn iterations to your from-scratch models!

---

### **STEP 4: Run the Training**

**Option A: Via Airflow (Recommended)**
1. Go to: `http://localhost:8080`
2. Find: `model_training_pipeline`
3. Click: **"Trigger DAG"**
4. Wait for it to complete

**Option B: Direct Run (for testing)**
```bash
cd /path/to/customer-churn-mlops
python -m src.classification.retail_classification_benchmark
```

---

### **STEP 5: Check the Results**

You'll see output like this:
```
════════════════════════════════════════════════════════════
📊 LogisticRegression Classification
════════════════════════════════════════════════════════════
   Epoch 0/100, Loss: 0.6931
   Epoch 10/100, Loss: 0.4567
   ...
   💾 Saved to MinIO

   📈 Results:
      Accuracy:  0.8524
      Precision: 0.7834
      Recall:    0.8123
      F1 Score:  0.7976
```

**Keep training until your models perform well!**

---

### **STEP 6: Download Models from MinIO**

Once your models are good enough:

1. Open MinIO: `http://localhost:9001`
2. Login:
   - Username: `minio_admin`
   - Password: `minio_password`
3. Go to bucket: **`models`**
4. Download ALL files ending with: `_classification_latest.pkl`

You should have **12 files total** (11 from-scratch + 1 sklearn)

---

### **STEP 7: Send Models to Me**

**📧 When your models are ready, send the `.pkl` files to me via:**
- **Discord** (preferred)
- **Gmail** (alternative)

**Subject/Message:** "Classification Models Ready - [Your Name]"

**Attach:** All `.pkl` files from MinIO

---

## ✅ Final Checklist

Before sending:

- [ ] Built all 11 from-scratch models
- [ ] All models inherit from `BaseModel`
- [ ] Imported models in `retail_classification_benchmark.py`
- [ ] Configured `MODEL_CONFIG` with hyperparameters
- [ ] Added all models to the `models` list
- [ ] Configured sklearn model with **matching iterations**
- [ ] Ran training via Airflow
- [ ] Models perform reasonably well (F1 score > 0.70 is good)
- [ ] Downloaded all `.pkl` files from MinIO
- [ ] Ready to send via Discord/Gmail

---

## 🎓 Important Info

**Target Variable:**
- `churned` (0 = customer stayed, 1 = customer left)
- Binary classification problem

**Evaluation Metrics:**
- **F1 Score** is the most important (balances precision & recall)
- Also check: Accuracy, Precision, Recall

**If Something Breaks:**
1. Check the model's `fit()` method matches the base class
2. Make sure `predict()` returns 0 or 1 (not probabilities)
3. Look at regression model examples
4. Check Airflow logs at `http://localhost:8080`

---

## 🚀 You Got This!

If you have questions, reach out. Otherwise, follow these steps and send me the models when they're ready!

**Good luck! 💪**
