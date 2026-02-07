# Feature Analysis & Modeling Strategy

**Role**: Lead Data Scientist  
**Purpose**: Strategic analysis and recommendations for feature engineering and modeling  
**Date**: February 2026

---

## Table of Contents
1. [Current Feature Analysis](#current-feature-analysis)
2. [Correlation Analysis](#correlation-analysis)
3. [Class Imbalance Strategy](#class-imbalance-strategy)
4. [V2 Feature Engineering](#v2-feature-engineering)
5. [Modeling Recommendations](#modeling-recommendations)

---

## Current Feature Analysis

### Feature Inventory (V1)

Based on [feature_engineering.py](file:///Users/jul/Desktop/uni/customer-churn-mlops/src/lead_ds/feature_engineering.py) and EDA results:

| Feature | Type | Source | Cardinality | Notes |
|---------|------|--------|-------------|-------|
| **Traffic Source** | Categorical | `utm_source` | 3 (gsearch, socialbook, direct) | One-hot encoded |
| **Device Type** | Categorical | `device_type` | 2 (mobile, desktop) | One-hot encoded |
| **Is Repeat Session** | Binary | `is_repeat_session` | 2 (0, 1) | Existing in raw data |
| **Hour of Day** | Numerical | `created_at` | 24 (0-23) | Engineered from timestamp |
| **Is Weekend** | Binary | `created_at` | 2 (0, 1) | Engineered from timestamp |
| **Landing Page** | Categorical | `pageview_url` | 7 (/home, /lander-1 to /lander-5) | One-hot encoded |
| **Engagement Depth** | Numerical | `website_pageviews` | 1-50+ | Page views per session |

**After One-Hot Encoding**: 14 total features

### Target Variables

| Target | Type | Distribution | Use Case |
|--------|------|--------------|----------|
| **is_ordered** | Binary | 6.8% positive, 93.2% negative | Classification (conversion prediction) |
| **revenue** | Continuous | $0-$150, AOV = $59.99 | Regression (revenue prediction) |

---

## Correlation Analysis

### Key Findings from EDA

Based on [07_correlation_matrix.png](file:///Users/jul/Desktop/uni/customer-churn-mlops/reports/figures/07_correlation_matrix.png):

#### 🔥 Strong Predictors (Correlation with `is_ordered`)

1. **Engagement Depth** - **STRONGEST PREDICTOR**
   - **Correlation**: High positive correlation
   - **Evidence**: 
     - 1 page (bounce) → 0.0% conversion
     - 2 pages → ~10% conversion
     - 5+ pages → **50.1% conversion**
   - **Insight**: Engagement is THE key driver
   - **Recommendation**: This will dominate tree-based models' feature importance

2. **Is Repeat Session**
   - **Correlation**: Moderate positive
   - **Evidence**:
     - New users: 6.64% conversion
     - Repeat users: 7.83% conversion (+18% lift)
   - **Insight**: User loyalty matters

3. **Traffic Source (Direct)**
   - **Correlation**: Weak positive
   - **Evidence**:
     - Direct: 7.28% conversion (best)
     - Organic search: varies
   - **Insight**: High-intent traffic converts better

#### ⚠️ Weak Predictors

1. **Hour of Day**
   - **Correlation**: Very weak
   - **Evidence**: 
     - Peak hour (14:00): 7.26% conversion
     - Only 0.5% lift over average
   - **Insight**: Non-linear relationship possible (afternoon spike)
   - **Recommendation**: May need binning (e.g., "peak hours" flag)

2. **Is Weekend**
   - **Correlation**: Near zero
   - **Evidence**:
     - Weekend: 6.77% conversion
     - Weekday: 6.84% conversion
   - **Insight**: Minimal impact
   - **Recommendation**: Consider dropping or combining with hour

3. **Device Type**
   - **Correlation**: Weak (context-dependent)
   - **Insight**: May have interaction effects with landing page

4. **Landing Page**
   - **Correlation**: Varies by page
   - **Insight**: A/B test results - some landers perform better
   - **Recommendation**: Keep for personalization insights

### Feature Importance Prediction

**Expected Ranking** (for tree-based models):

```
1. engagement_depth          ████████████████████ (90%)
2. is_repeat_session         ████████ (40%)
3. utm_source_direct         ████ (20%)
4. landing_page_*            ███ (15%)
5. hour_of_day               ██ (10%)
6. device_type_mobile        ██ (10%)
7. is_weekend                █ (5%)
```

### Correlation with Revenue

For regression models:
- **Engagement depth**: Strong positive (more pages → higher likelihood of purchase → revenue)
- **Revenue distribution**: Tight around $60 (fixed prices)
- **Strategy**: Predict `is_ordered` first, then predict revenue (two-stage model)

---

## Class Imbalance Strategy

### The Problem

**Distribution**:
- ❌ No Purchase: 440,558 sessions (93.2%)
- ✅ Purchase: 32,313 sessions (6.8%)

**Imbalance Ratio**: 13.6:1 (severe imbalance)

### Why Imbalance is Critical

1. **Naive Baseline**: Model predicting "no purchase" for all sessions achieves 93.2% accuracy
2. **Metric Failure**: Accuracy is meaningless
3. **Learning Bias**: Model learns to ignore minority class
4. **Business Impact**: We CARE about the 6.8% who convert!

---

### Strategy 1: Evaluation Metrics (CRITICAL)

**❌ Don't Use**:
- Accuracy (misleading - 93% baseline)

**✅ Use Instead**:

| Metric | Formula | Why | Target |
|--------|---------|-----|--------|
| **F1-Score** | 2 × (precision × recall) / (precision + recall) | Balances false positives and false negatives | > 0.40 |
| **AUC-ROC** | Area under ROC curve | Measures separation of classes | > 0.75 |
| **Precision** | TP / (TP + FP) | "Of predicted converters, how many actually converted?" | > 0.30 |
| **Recall** | TP / (TP + FN) | "Of actual converters, how many did we catch?" | > 0.50 |
| **Precision-Recall Curve** | Trade-off visualization | Better than ROC for imbalanced data | Visual |

**Primary Metric**: **F1-Score** (balances precision and recall)

---

### Strategy 2: Sampling Techniques

#### Option A: SMOTE (Synthetic Minority Over-sampling Technique)

**How it works**: Generates synthetic examples of minority class

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Original: 93.2% / 6.8%
# After SMOTE: 70% / 30% (more balanced)
```

**Pros**:
- ✅ Increases minority class representation
- ✅ Reduces model bias toward majority class
- ✅ Works well with tree-based models

**Cons**:
- ❌ Can overfit to minority class
- ❌ Increases training time
- ❌ May create unrealistic synthetic examples

**Recommendation**: **Use SMOTE with `sampling_strategy=0.2` to 0.3`** (don't fully balance - keep some imbalance)

---

#### Option B: Random Under-sampling

**How it works**: Randomly removes majority class samples

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Pros**:
- ✅ Fast training (smaller dataset)
- ✅ Reduces majority class dominance

**Cons**:
- ❌ Loses information (discards 80%+ of data)
- ❌ Not recommended with only 472K samples

**Recommendation**: **Avoid** - we don't have enough data to throw away

---

#### Option C: Hybrid (SMOTE + Tomek Links)

**How it works**: SMOTE + clean overlapping samples

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

**Recommendation**: **Good option** for reducing noise after SMOTE

---

### Strategy 3: Class Weights

**How it works**: Penalize misclassifications of minority class more heavily

```python
from sklearn.ensemble import RandomForestClassifier

# Option 1: Automatic balancing
model = RandomForestClassifier(class_weight='balanced')

# Option 2: Custom weights
class_weights = {0: 1, 1: 13.6}  # Based on imbalance ratio
model = RandomForestClassifier(class_weight=class_weights)
```

**Pros**:
- ✅ No data modification (keeps original dataset)
- ✅ Fast - no resampling overhead
- ✅ Supported by most sklearn models

**Cons**:
- ❌ Less effective than SMOTE for extreme imbalance
- ❌ Requires hyperparameter tuning of weights

**Recommendation**: **Use as baseline** - always try `class_weight='balanced'` first

---

### Strategy 4: Threshold Adjustment

**How it works**: Adjust decision threshold from default 0.5

```python
# Default: predict class 1 if probability > 0.5
# Adjusted: predict class 1 if probability > 0.3 (more lenient)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)  # Lower threshold
```

**Use case**: Increase recall at the cost of precision

**Recommendation**: **Tune threshold** after training to optimize F1-score

---

### Strategy 5: Ensemble Methods

**Recommendation**: Use algorithms that handle imbalance well:

| Algorithm | Imbalance Handling | Recommendation |
|-----------|-------------------|----------------|
| **XGBoost** | `scale_pos_weight` parameter | ⭐⭐⭐⭐⭐ Best choice |
| **LightGBM** | `is_unbalance=True` | ⭐⭐⭐⭐⭐ Fast & effective |
| **Random Forest** | `class_weight='balanced'` | ⭐⭐⭐⭐ Good baseline |
| **Logistic Regression** | `class_weight='balanced'` | ⭐⭐⭐ Baseline only |

**XGBoost Example**:
```python
import xgboost as xgb

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~13.6

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    eval_metric='aucpr'  # Precision-Recall AUC
)
```

---

### Recommended Approach (Best Practice)

**Phase 1: Baseline (Week 1)**
```python
# 1. Use class weights (no SMOTE)
# 2. Random Forest with class_weight='balanced'
# 3. Evaluate with F1-score and AUC-ROC
# 4. Establish performance baseline
```

**Phase 2: SMOTE (Week 2)**
```python
# 1. Apply SMOTE with sampling_strategy=0.2
# 2. Random Forest (no class weights needed)
# 3. Compare to baseline
# 4. Check for overfitting on test set
```

**Phase 3: Advanced (Week 3)**
```python
# 1. XGBoost with scale_pos_weight
# 2. Threshold tuning
# 3. Hyperparameter optimization
# 4. Final model selection
```

---

## V2 Feature Engineering

### High-Priority Features

Based on EDA insights and business logic:

#### 1. Session Duration ⭐⭐⭐⭐⭐

**Calculation**:
```python
# Time between first and last pageview
session_duration = df_pageviews.groupby('website_session_id').apply(
    lambda x: (x['created_at'].max() - x['created_at'].min()).total_seconds()
)
```

**Hypothesis**: Longer sessions → higher engagement → higher conversion

**Expected Impact**: HIGH (correlates with engagement_depth)

**Example**:
- 10 seconds (bounce) → 0% conversion
- 300 seconds (5 minutes) → 30% conversion
- 600+ seconds (10+ minutes) → 50% conversion

---

#### 2. Bounce Rate Indicator ⭐⭐⭐⭐⭐

**Calculation**:
```python
# Session with only 1 pageview
is_bounce = (engagement_depth == 1).astype(int)
```

**Hypothesis**: Bounces never convert (confirmed by EDA)

**Expected Impact**: HIGH (strong negative predictor)

**Evidence from EDA**: 1-page sessions have 0.0% conversion rate

---

#### 3. Pages Per Minute (Engagement Velocity) ⭐⭐⭐⭐

**Calculation**:
```python
# Engagement depth per minute
pages_per_minute = engagement_depth / (session_duration / 60)
```

**Hypothesis**: Fast browsing (high velocity) → exploratory vs. slow browsing → deliberate

**Trade-offs**:
- High velocity: User exploring quickly (good)
- Low velocity: User reading carefully (also good?)
- Very high velocity: Bot or accidental clicks (bad)

**Expected Impact**: MEDIUM (non-linear relationship)

**Recommendation**: Create bins:
- `< 0.5 ppm`: Slow/deliberate
- `0.5-2 ppm`: Normal
- `> 2 ppm`: Very fast (possible bot)

---

#### 4. UTM Campaign Granularity ⭐⭐⭐⭐

**Current**: Only `utm_source` (gsearch, socialbook, direct)  
**V2**: Add `utm_campaign` and `utm_content`

**Calculation**:
```python
# Already in raw data
# Use top 10 campaigns, bin others as "other"
top_campaigns = df['utm_campaign'].value_counts().head(10).index
df['utm_campaign_grouped'] = df['utm_campaign'].apply(
    lambda x: x if x in top_campaigns else 'other'
)
```

**Hypothesis**: Different campaigns have different conversion rates

**Expected Impact**: MEDIUM-HIGH (captures marketing effectiveness)

**Business Value**: Identify best-performing campaigns

---

#### 5. Device × Traffic Source Interaction ⭐⭐⭐

**Calculation**:
```python
# Combine device and traffic source
df['device_source'] = df['device_type'] + '_' + df['utm_source']
# Example: "mobile_gsearch", "desktop_direct"
```

**Hypothesis**: Conversion varies by device-source combination

**Example**:
- Desktop + Direct → High conversion (high intent)
- Mobile + Social → Low conversion (browsing)

**Expected Impact**: MEDIUM (captures interaction effects)

**Recommendation**: One-hot encode (creates ~6 features)

---

#### 6. Time Since Last Visit (for Repeat Users) ⭐⭐⭐

**Calculation**:
```python
# For repeat users, calculate days since last session
user_sessions = df.sort_values('created_at').groupby('user_id')
df['days_since_last_visit'] = user_sessions['created_at'].diff().dt.days
df['days_since_last_visit'].fillna(-1, inplace=True)  # -1 for new users
```

**Hypothesis**: Optimal frequency exists (not too soon, not too late)

**Expected Pattern**:
- 0-3 days: High conversion (still interested)
- 4-14 days: Medium conversion
- 15+ days: Low conversion (lost interest)

**Expected Impact**: MEDIUM (for repeat users segment)

---

#### 7. Landing Page Category ⭐⭐

**Calculation**:
```python
# Group landing pages by type
def categorize_landing_page(page):
    if page == '/home':
        return 'homepage'
    elif 'lander' in page:
        return 'landing_page'
    elif 'product' in page:
        return 'product_page'
    else:
        return 'other'

df['landing_page_category'] = df['landing_page'].apply(categorize_landing_page)
```

**Hypothesis**: Landing page type affects conversion

**Expected Impact**: MEDIUM (aggregates A/B test variants)

---

#### 8. Hour Bins (Instead of Raw Hour) ⭐⭐

**Calculation**:
```python
def bin_hour(hour):
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'  # Peak conversion
    else:
        return 'evening'

df['hour_bin'] = df['hour_of_day'].apply(bin_hour)
```

**Reason**: Hour has non-linear relationship with conversion

**Evidence**: Peak at 14:00 (2 PM), but raw hour doesn't capture this well

**Expected Impact**: MEDIUM (better than raw hour)

---

#### 9. Pageview Depth Bins ⭐⭐⭐

**Calculation**:
```python
def bin_engagement(depth):
    if depth == 1:
        return 'bounce'
    elif depth <= 3:
        return 'low'
    elif depth <= 6:
        return 'medium'
    else:
        return 'high'

df['engagement_bin'] = df['engagement_depth'].apply(bin_engagement)
```

**Reason**: Engagement has non-linear relationship (0% → 10% → 30% → 50%)

**Expected Impact**: MEDIUM (captures thresholds)

**Alternative**: Keep raw `engagement_depth` for tree models (they bin automatically)

---

#### 10. HTTP Referer Category ⭐⭐

**Calculation**:
```python
def categorize_referer(referer):
    if pd.isna(referer) or referer == '':
        return 'direct_or_unknown'
    elif 'google' in referer.lower():
        return 'google'
    elif 'facebook' in referer.lower() or 'social' in referer.lower():
        return 'social'
    else:
        return 'other'

df['referer_category'] = df['http_referer'].apply(categorize_referer)
```

**Hypothesis**: Referer provides additional traffic source context

**Expected Impact**: LOW-MEDIUM (overlaps with utm_source)

---

### Feature Engineering Roadmap

**Priority 1 (High Impact, Easy Implementation)**:
1. ✅ Bounce rate indicator (`is_bounce`)
2. ✅ Session duration
3. ✅ Pages per minute
4. ✅ Hour bins (instead of raw hour)

**Priority 2 (Medium Impact, Moderate Effort)**:
5. ⭐ UTM campaign granularity
6. ⭐ Device × Source interaction
7. ⭐ Time since last visit (repeat users)

**Priority 3 (Exploration)**:
8. 🔬 Landing page category
9. 🔬 Referer category

---

### Implementation Steps

**Step 1: Update `feature_engineering.py`**

Add new features to the feature engineering pipeline:
```python
# In src/lead_ds/feature_engineering.py

# Session duration
pageview_times = df_pageviews.groupby('website_session_id')['created_at'].agg(['min', 'max'])
pageview_times['session_duration'] = (pageview_times['max'] - pageview_times['min']).dt.total_seconds()
df_sessions = df_sessions.merge(pageview_times[['session_duration']], ...)

# Bounce rate
df_sessions['is_bounce'] = (df_sessions['engagement_depth'] == 1).astype(int)

# Pages per minute
df_sessions['pages_per_minute'] = df_sessions['engagement_depth'] / (df_sessions['session_duration'] / 60 + 0.01)
```

**Step 2: Re-run EDA**

Analyze new features:
- Correlation with targets
- Distribution analysis
- Feature importance in baseline models

**Step 3: A/B Test V1 vs V2**

Compare model performance:
- V1 (current 7 features): Baseline F1-score
- V2 (with new features): Improved F1-score?

**Step 4: Feature Selection**

Remove low-impact features:
- Check feature importance from Random Forest
- Drop features with < 5% importance
- Reduce multicollinearity

---

## Modeling Recommendations

### Classification (Conversion Prediction)

**Recommended Pipeline**:

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

# Step 1: Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.25)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 2: Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',  # Extra insurance
    random_state=42,
    n_jobs=-1
)
model.fit(X_resampled, y_resampled)

# Step 3: Evaluate on ORIGINAL test set (not resampled!)
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

**Performance Targets**:
- F1-Score: > 0.40
- AUC-ROC: > 0.75
- Recall: > 0.50 (catch at least half of converters)

---

### Regression (Revenue Prediction)

**Recommended Strategy**: Two-Stage Model

**Stage 1: Predict Conversion (Classification)**
```python
# Use classification model to predict is_ordered
y_conversion_pred = classification_model.predict(X)
```

**Stage 2: Predict Revenue (Regression on Converting Sessions)**
```python
# Train regression model only on converting sessions
X_converting = X_train[y_train_conversion == 1]
y_revenue = revenue_train[y_train_conversion == 1]

regression_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42
)
regression_model.fit(X_converting, y_revenue)

# Predict revenue for predicted converters
revenue_pred = np.zeros(len(X_test))
converter_mask = (y_conversion_pred == 1)
revenue_pred[converter_mask] = regression_model.predict(X_test[converter_mask])
```

**Performance Targets**:
- RMSE: < $20 (AOV = $60)
- MAE: < $15
- R²: > 0.50

---

## Summary & Next Steps

### Current State
- ✅ 7 features engineered (V1)
- ✅ EDA completed with insights
- ✅ Correlation analysis shows **engagement_depth** is key predictor
- ⚠️ Severe class imbalance (93.2% / 6.8%)

### Recommendations

**Immediate (Week 1)**:
1. Implement baseline model with `class_weight='balanced'`
2. Use F1-score as primary metric
3. Establish baseline performance

**Short-term (Week 2)**:
1. Implement SMOTE with `sampling_strategy=0.2-0.3`
2. Add V2 Priority 1 features (bounce, duration, pages_per_minute)
3. Re-run EDA on new features

**Medium-term (Week 3)**:
1. Upgrade to XGBoost with `scale_pos_weight`
2. Hyperparameter tuning
3. Feature selection (drop low-importance features)

**Long-term**:
1. Implement two-stage model for revenue prediction
2. A/B test model in production
3. Monitor for data drift
4. Continuous feature engineering

---

**The path to 0.40+ F1-score is clear: Handle imbalance with SMOTE, leverage engagement_depth, and iterate on features! 🚀**
