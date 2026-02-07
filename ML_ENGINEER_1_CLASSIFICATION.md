# ML Engineer #1: Classification Model - Task Assignment

**Your Mission**: Implement a classification model to predict customer conversion (`is_ordered`)

**Timeline**: 1-3 weeks  
**Performance Target**: F1-Score > 0.40, AUC-ROC > 0.75

---

## 🎯 Your Objective

Predict whether a website session will result in a purchase.

**Target Variable**: `is_ordered` (binary: 0 = no purchase, 1 = purchase)

**Challenge**: **Severe class imbalance** - 93.2% don't buy, only 6.8% buy

---

## 📁 Your File

**Location**: [src/classification/train_model.py](file:///Users/jul/Desktop/uni/customer-churn-mlops/src/classification/train_model.py)

**What's Already Done**:
- ✅ Data loading from MinIO
- ✅ Train/test split (80/20)
- ✅ MLflow experiment tracking
- ✅ Model evaluation (F1, AUC-ROC)
- ✅ Model saving to MinIO

**What YOU Need to Do**:
- ⚠️ Implement the `train_model()` function (line 68-103)

---

## 🔧 Step-by-Step Instructions

### Step 1: Open Your File

```bash
cd /Users/jul/Desktop/uni/customer-churn-mlops
code src/classification/train_model.py
```

### Step 2: Find the Function to Implement

**Line 68-103**: Look for this function:

```python
def train_model(self, X_train, y_train):
    """
    🚨 ML ENGINEERS: IMPLEMENT YOUR MODEL HERE
    ...
    """
    
    # PLACEHOLDER - Replace with your implementation
    print("\n⚠️  PLACEHOLDER MODEL - Replace with actual implementation")
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy='most_frequent')
    model.fit(X_train, y_train)
    
    return model
```

**Your Task**: Replace the `DummyClassifier` with a real model.

---

## 📊 The Data

**Training Data**: `s3://processed-data/training_data.parquet`

**Features** (X_train has 12 features):
- `is_repeat_session` (binary)
- `hour_of_day` (0-23)
- `is_weekend` (binary)
- `engagement_depth` (number of pages viewed) ⭐ **STRONGEST PREDICTOR**
- `utm_source_*` (one-hot encoded: gsearch, socialbook)
- `device_type_mobile` (one-hot encoded)
- `landing_page_*` (one-hot encoded: /lander-1 through /lander-5)

**Target** (y_train):
- `is_ordered`: 0 (no purchase) or 1 (purchase)

**Distribution**:
- Training set: 378,296 samples
- Test set: 94,575 samples
- Class balance: 93.2% negative, 6.8% positive (IMBALANCED!)

---

## 🚨 CRITICAL: Handle Class Imbalance

**You MUST address the 93.2% / 6.8% imbalance or your model will be useless.**

### Strategy 1: SMOTE (Recommended)

```python
def train_model(self, X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    
    # Step 1: Apply SMOTE to balance classes
    print("   Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42, sampling_strategy=0.25)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"   Original distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"   After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # Step 2: Train model
    print("   Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',  # Extra protection
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_resampled, y_resampled)
    
    return model
```

### Strategy 2: Class Weights (Baseline)

```python
def train_model(self, X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    
    print("   Training Random Forest with class_weight='balanced'...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',  # Handles imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    return model
```

### Strategy 3: XGBoost (Advanced)

```python
def train_model(self, X_train, y_train):
    import xgboost as xgb
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    print("   Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        eval_metric='aucpr',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model
```

---

## ✅ Testing Your Model

### Test Locally (Before Airflow)

```bash
# Run your model
python src/classification/train_model.py
```

**Expected Output**:
```
============================================================
  CLASSIFICATION MODEL TRAINING
  Task: Conversion Prediction (is_ordered)
============================================================
📥 Loading training data...
✅ Loaded 472,871 samples
   Features: 12
   Class distribution: {0: 440558, 1: 32313}

🔀 Splitting data (test_size=0.2)...
   Train: 378,296 samples
   Test:  94,575 samples

   Applying SMOTE to handle class imbalance...
   Training Random Forest Classifier...

📊 Evaluating model...

📈 Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.85      0.90     88112
           1       0.25      0.55      0.35      6463

    accuracy                           0.83     94575
   macro avg       0.61      0.70      0.62     94575
weighted avg       0.91      0.83      0.86     94575


🎯 Key Metrics:
   F1-Score: 0.35
   AUC-ROC:  0.78

💾 Saving model to MinIO...
✅ Model saved to s3://models/classification_model.pkl

============================================================
✅ TRAINING COMPLETE!
   MLflow Run ID: abc123...
   F1-Score: 0.35
   AUC-ROC: 0.78
============================================================
```

### Check MLflow

```bash
# Open MLflow UI
open http://localhost:5001
```

Navigate to:
- Experiment: `Conversion_Prediction_Classification`
- Check your run's metrics:
  - `f1_score`
  - `auc_roc`
  - Parameters logged

### Test in Airflow

```bash
# Trigger the model training DAG
airflow dags trigger model_training_pipeline
```

Check Airflow UI at `http://localhost:8080`:
- DAG: `model_training_pipeline`
- Task: `train_classification_model`
- View logs

---

## 🎯 Performance Targets

| Metric | Minimum Target | Good Target | Notes |
|--------|---------------|-------------|-------|
| **F1-Score** | > 0.30 | > 0.40 | Primary metric - balances precision & recall |
| **AUC-ROC** | > 0.70 | > 0.75 | Measures class separation |
| **Recall** | > 0.40 | > 0.50 | Catch at least 50% of converters |
| **Precision** | > 0.20 | > 0.30 | Reduce false positives |

**⚠️ DO NOT optimize for accuracy** - it's meaningless with imbalanced data!

---

## 📚 Algorithm Recommendations

### Phase 1: Baseline (Start Here)
```python
# Random Forest with class_weight='balanced'
# Expected F1: 0.25-0.35
```

### Phase 2: SMOTE
```python
# SMOTE + Random Forest
# Expected F1: 0.35-0.42
```

### Phase 3: Advanced
```python
# XGBoost with scale_pos_weight
# Expected F1: 0.40-0.50
```

---

## 🔍 Key Insights from EDA

**Use these to guide your modeling**:

1. **Engagement Depth = Strongest Predictor**
   - 1 page (bounce): 0% conversion
   - 5+ pages: 50% conversion
   - This feature will dominate your feature importance

2. **Repeat Users Convert More**
   - New users: 6.64%
   - Repeat users: 7.83% (+18% lift)

3. **Traffic Source Matters**
   - Direct: 7.28% conversion (best)
   - Organic search: lower

4. **Time Patterns are Weak**
   - Peak hour (2 PM): 7.26%
   - Only 0.5% lift - don't expect much from time features

**Read More**: [FEATURE_ANALYSIS_STRATEGY.md](file:///Users/jul/Desktop/uni/customer-churn-mlops/FEATURE_ANALYSIS_STRATEGY.md)

---

## 🛠️ Troubleshooting

### Issue: F1-Score = 0.00
**Cause**: Model predicts all negative class (ignoring minority class)  
**Fix**: Apply SMOTE or increase class weights

### Issue: High Recall, Low Precision
**Cause**: Model predicts too many positives (lots of false positives)  
**Fix**: Adjust decision threshold or reduce SMOTE sampling_strategy

### Issue: Model Takes Too Long to Train
**Cause**: SMOTE creates too many samples  
**Fix**: Use `sampling_strategy=0.2` instead of 0.5, or reduce `n_estimators`

### Issue: MLflow Logging Fails
**Cause**: MLflow server not running  
**Fix**: Start MLflow server
```bash
mlflow server --host 0.0.0.0 --port 5001
```

---

## 📦 Required Libraries

Already installed in the environment:
- `scikit-learn` - Random Forest, Logistic Regression
- `imbalanced-learn` - SMOTE
- `xgboost` - XGBoost (if you use it)
- `mlflow` - Experiment tracking
- `pandas`, `numpy` - Data manipulation

If you need additional libraries:
```bash
pip install lightgbm  # For LightGBM (optional)
```

---

## 📝 Deliverables Checklist

Before you consider yourself "done":

- [ ] Implemented `train_model()` function
- [ ] Handled class imbalance (SMOTE or class weights)
- [ ] Achieved F1-score > 0.30 (minimum)
- [ ] Tested locally: `python src/classification/train_model.py`
- [ ] Verified MLflow logging works
- [ ] Model saved to MinIO successfully
- [ ] Tested in Airflow DAG
- [ ] Documented your approach (add comments in code)
- [ ] Tried at least 2 different algorithms
- [ ] Performed hyperparameter tuning (optional but recommended)

---

## 🚀 Next Steps After Completion

Once your model is working:

1. **Feature Importance Analysis**
   ```python
   import matplotlib.pyplot as plt
   
   # Get feature importance
   importance = model.feature_importances_
   features = X_train.columns
   
   # Plot
   plt.barh(features, importance)
   plt.xlabel('Importance')
   plt.title('Feature Importance')
   plt.tight_layout()
   plt.savefig('reports/figures/feature_importance.png')
   ```

2. **Threshold Tuning**
   ```python
   from sklearn.metrics import precision_recall_curve
   
   # Find optimal threshold
   precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
   f1_scores = 2 * (precision * recall) / (precision + recall)
   best_threshold = thresholds[np.argmax(f1_scores)]
   
   print(f"Best threshold: {best_threshold:.3f}")
   ```

3. **Model Comparison**
   - Compare Random Forest vs XGBoost vs LightGBM
   - Use MLflow to track all experiments
   - Select best model based on F1-score

---

## 📞 Need Help?

1. **Data Questions**: Check [FEATURE_ANALYSIS_STRATEGY.md](file:///Users/jul/Desktop/uni/customer-churn-mlops/FEATURE_ANALYSIS_STRATEGY.md)
2. **Infrastructure Issues**: Contact Lead Engineer
3. **Algorithm Advice**: Check [ML_ENGINEER_HANDOFF.md](file:///Users/jul/Desktop/uni/customer-churn-mlops/ML_ENGINEER_HANDOFF.md)
4. **Class Imbalance**: See detailed strategies in [FEATURE_ANALYSIS_STRATEGY.md](file:///Users/jul/Desktop/uni/customer-churn-mlops/FEATURE_ANALYSIS_STRATEGY.md) (Section: Class Imbalance Strategy)

---

## 📚 Resources

**Class Imbalance**:
- [Imbalanced-learn Docs](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)

**Algorithms**:
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)

**Metrics**:
- [F1-Score Explained](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

---

**Good luck! Remember: Handle the imbalance, leverage engagement_depth, and aim for F1 > 0.40! 🚀**
