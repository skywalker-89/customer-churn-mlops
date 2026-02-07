# ML Engineer Handoff Document

## 🎯 Your Mission

You have **2 Machine Learning Engineers** who need to implement:
1. **Classification Model**: Predict if a user will convert (`is_ordered`)
2. **Regression Model**: Predict revenue (`revenue`)

## 📁 What's Already Built (Infrastructure)

### ✅ Data Pipeline
- [x] **Training Data**: `s3://processed-data/training_data.parquet`
  - 472,871 sessions
  - 14 features (one-hot encoded)
  - 2 targets: `is_ordered`, `revenue`

### ✅ Model Training Templates
- [x] [src/classification/train_model.py](file:///Users/jul/Desktop/uni/customer-churn-mlops/src/classification/train_model.py)
- [x] [src/regression/train_model.py](file:///Users/jul/Desktop/uni/customer-churn-mlops/src/regression/train_model.py)

These templates include:
- Data loading from MinIO ✅
- Train/test split ✅
- MLflow experiment tracking ✅
- Model saving to MinIO ✅

### ✅ Orchestration
- [x] [dags/model_training_dag.py](file:///Users/jul/Desktop/uni/customer-churn-mlops/dags/model_training_dag.py)
  - Validates training data
  - Runs both models in parallel
  - Integrates with Airflow

---

## 🚧 What ML Engineers Need to Implement

### Task 1: Classification Model

**File**: `src/classification/train_model.py`  
**Function to implement**: `train_model(self, X_train, y_train)`

#### Requirements

1. **Handle Class Imbalance** (CRITICAL)
   - Problem: 93.2% don't buy, 6.8% buy
   - Solutions:
     ```python
     # Option A: SMOTE
     from imblearn.over_sampling import SMOTE
     smote = SMOTE(random_state=42)
     X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
     
     # Option B: Class weights
     model = RandomForestClassifier(class_weight='balanced')
     
     # Option C: Adjust decision threshold
     y_pred = (y_proba > 0.3).astype(int)  # Lower threshold
     ```

2. **Choose Algorithm**
   - **Baseline**: Logistic Regression
   - **Recommended**: Random Forest or XGBoost
   - **Advanced**: LightGBM, CatBoost

3. **Optimize for F1-Score** (NOT accuracy)
   - Accuracy is misleading (93% by always predicting "no purchase")
   - Use F1-score or AUC-ROC

4. **Hyperparameter Tuning**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_estimators': [100, 200],
       'max_depth': [10, 15, 20],
       'min_samples_split': [2, 5, 10]
   }
   
   grid = GridSearchCV(model, param_grid, cv=5, scoring='f1')
   grid.fit(X_train, y_train)
   ```

#### Example Implementation

```python
def train_model(self, X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    
    # Handle imbalance
    print("   Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Train model
    print("   Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_resampled, y_resampled)
    
    return model
```

---

### Task 2: Regression Model

**File**: `src/regression/train_model.py`  
**Function to implement**: `train_model(self, X_train, y_train)`

#### Requirements

1. **Choose Regression Strategy**

   **Option A: Two-Stage Model** (Recommended)
   ```python
   # Stage 1: Predict if user will convert (classification)
   # Stage 2: If yes, predict revenue (regression)
   
   # This separates the problems:
   # - First model handles 93/7 imbalance
   # - Second model only predicts order value for converters
   ```

   **Option B: Direct Regression**
   ```python
   # Predict revenue for all sessions (most are $0)
   # Needs to handle zero-inflation
   ```

   **Option C: Regression on Converting Sessions Only** (Simplest)
   ```python
   # Only train on sessions where is_ordered == 1
   # Template already supports this: strategy='converting_only'
   ```

2. **Handle Distribution**
   - Revenue is long-tailed (most orders ~$60, some higher)
   - Consider log transformation:
     ```python
     y_log = np.log1p(y_train)  # log(1 + y)
     model.fit(X_train, y_log)
     
     # Predictions
     y_pred = np.expm1(model.predict(X_test))  # exp(y) - 1
     ```

3. **Choose Algorithm**
   - **Baseline**: Linear Regression
   - **Recommended**: Random Forest Regressor
   - **Advanced**: XGBoost Regressor, LightGBM

4. **Optimize for RMSE or MAE**

#### Example Implementation

```python
def train_model(self, X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Optional: Log transform
    y_log = np.log1p(y_train)
    
    # Train model
    print("   Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_log)
    
    # Wrap model to inverse transform predictions
    class LogTransformWrapper:
        def __init__(self, model):
            self.model = model
        
        def predict(self, X):
            return np.expm1(self.model.predict(X))
        
        def __getattr__(self, attr):
            return getattr(self.model, attr)
    
    return LogTransformWrapper(model)
```

---

## 🧪 How to Test Your Models

### Local Testing (Before Airflow)

```bash
# Test classification model
python src/classification/train_model.py

# Test regression model
python src/regression/train_model.py
```

### Check MLflow

```bash
# MLflow should be running on http://localhost:5001
# View your experiments:
# 1. Conversion_Prediction_Classification
# 2. Revenue_Prediction_Regression
```

### Run in Airflow

```bash
# Trigger the DAG
airflow dags trigger model_training_pipeline

# Or use the Airflow UI
# Navigate to http://localhost:8080
```

---

## 📊 Expected Performance Targets

Based on the EDA, here are realistic targets:

### Classification (Conversion Prediction)

| Metric | Target | Why |
|--------|--------|-----|
| **F1-Score** | > 0.40 | Balance precision and recall |
| **AUC-ROC** | > 0.75 | Good separation of classes |
| **Recall** | > 0.50 | Catch at least half of converters |
| **Precision** | > 0.30 | Reduce false positives |

⚠️ **Don't optimize for accuracy** - it will be ~93% by always predicting "no purchase"

### Regression (Revenue Prediction)

| Metric | Target | Why |
|--------|--------|-----|
| **RMSE** | < $20 | Average Order Value is $60 |
| **MAE** | < $15 | Most orders are $50-70 |
| **R²** | > 0.50 | Explain at least half the variance |
| **MAPE** | < 30% | Within 30% of actual revenue |

---

## 🎯 Key Insights from EDA (Use These!)

1. **Engagement is THE key predictor**
   - 5+ pages viewed → 50% conversion rate
   - 1 page (bounce) → 0% conversion
   - Feature importance: `engagement_depth` will dominate

2. **Repeat users convert more**
   - New: 6.64%
   - Repeat: 7.83% (+18% lift)

3. **Time matters**
   - Peak hour: 2 PM (7.26% conversion)
   - Weekend vs weekday: No significant difference

4. **Traffic source quality varies**
   - Direct traffic: 7.28% conversion (best)
   - Consider this in your model

---

## 📝 Deliverables Checklist

For each ML Engineer:

### ML Engineer 1: Classification
- [ ] Implement `train_model()` in `src/classification/train_model.py`
- [ ] Handle class imbalance (SMOTE or class weights)
- [ ] Achieve F1-score > 0.40
- [ ] Test locally: `python src/classification/train_model.py`
- [ ] Verify MLflow logging works
- [ ] Document approach and hyperparameters

### ML Engineer 2: Regression
- [ ] Implement `train_model()` in `src/regression/train_model.py`
- [ ] Choose regression strategy (recommend: converting_only)
- [ ] Achieve RMSE < $20
- [ ] Test locally: `python src/regression/train_model.py`
- [ ] Verify MLflow logging works
- [ ] Document approach and hyperparameters

---

## 🚀 Timeline Suggestion

**Week 1**: 
- ML Engineers implement baseline models (Logistic Regression, Linear Regression)
- Verify pipeline works end-to-end

**Week 2**:
- Implement advanced models (Random Forest, XGBoost)
- Hyperparameter tuning

**Week 3**:
- Optimize for metrics
- Document approach
- Final testing in Airflow

---

## 📞 Support

If ML Engineers need help:

1. **Data Questions**: Check the EDA report at `reports/figures/eda_summary_report.txt`
2. **Infrastructure Issues**: Contact Lead Engineer (you)
3. **Feature Engineering Ideas**: See walkthrough for suggestions
4. **MLflow Tracking**: http://localhost:5001

---

## 🎓 Resources

**Class Imbalance**:
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- SMOTE: Synthetic Minority Over-sampling Technique

**Model Selection**:
- Scikit-learn classifiers: `sklearn.ensemble`
- XGBoost: `xgboost.XGBClassifier` / `xgboost.XGBRegressor`
- LightGBM: `lightgbm.LGBMClassifier` / `lightgbm.LGBMRegressor`

**Hyperparameter Tuning**:
- `sklearn.model_selection.GridSearchCV`
- `sklearn.model_selection.RandomizedSearchCV`

**Metrics**:
- `sklearn.metrics.classification_report`
- `sklearn.metrics.roc_auc_score`
- `sklearn.metrics.mean_squared_error`

---

## ✅ You're All Set!

The infrastructure is ready. ML Engineers just need to implement the `train_model()` function in each template. Everything else (data loading, MLflow tracking, model saving) is handled for them.

**Good luck! 🚀**
