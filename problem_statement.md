# Problem Statement: Customer Churn Prediction and Sales Forecasting

## 🎯 Project Overview

This project addresses a critical business challenge in retail analytics: predicting customer behavior to optimize business strategies and reduce customer attrition. Using a comprehensive retail dataset, we implement both regression and classification models to solve two interconnected business problems.

## 📊 Dataset Description

**Dataset**: Retail Customer Analytics Dataset  
**Size**: 1,000 customer records with 70+ features  
**Source**: Synthetic retail transaction data  
**Business Context**: Multi-category retail store operations

### Key Features (70+ attributes across 6 categories):

1. **Customer Demographics** (5 features)

   - Age, gender, marital status, education level, occupation
   - Income bracket, number of children, zip code/city/state

2. **Customer Behavior** (8 features)

   - Loyalty program membership, membership years
   - Purchase frequency, average purchase value
   - Days since last purchase, total transactions
   - Online vs in-store purchase patterns

3. **Transaction History** (12 features)

   - Transaction dates, product categories, quantities
   - Unit prices, discount applications, payment methods
   - Store locations, transaction timing patterns

4. **Product Analytics** (15 features)

   - Product ratings, review counts, stock levels
   - Return rates, product characteristics (size, weight, color, material)
   - Shelf life, manufacture/expiry dates

5. **Marketing & Promotions** (8 features)

   - Promotion types, effectiveness metrics
   - Target audience segmentation, promotional channels
   - Seasonal and holiday patterns

6. **Customer Engagement** (6 features)
   - Customer support interactions, email subscriptions
   - App usage, website visits, social media engagement

## 🎯 Problem Definition

### Problem 1: Customer Churn Prediction (Classification)

**Target Variable**: `churned` (Binary: 0 = Retained, 1 = Churned)  
**Business Impact**: Customer retention is 5-25x cheaper than acquisition  
**Success Metric**: F1-Score (balances precision/recall for imbalanced data)

**Why This Matters**:

- Identify at-risk customers before they leave
- Enable proactive retention campaigns
- Reduce customer acquisition costs
- Maintain revenue stability

### Problem 2: Customer Lifetime Value Prediction (Regression)

**Target Variable**: `total_sales` (Continuous: Total customer spending)  
**Business Impact**: Optimize marketing spend allocation  
**Success Metric**: RMSE and R² (prediction accuracy and variance explained)

**Why This Matters**:

- Predict future revenue from customers
- Segment customers by value potential
- Optimize resource allocation
- Personalize marketing strategies

## 🏗️ Model Requirements

### Classification Models (11 from-scratch implementations):

1. **Logistic Regression** - Baseline binary classifier
2. **Decision Tree** - Interpretable rule-based model
3. **Random Forest** - Ensemble of decision trees
4. **Support Vector Machine** - Margin-based classifier
5. **Random Forest + PCA** - Dimensionality reduction variant
6. **SVM + PCA** - Feature reduction for SVM
7. **K-Means Clustering** - Unsupervised customer segmentation
8. **Agglomerative Clustering** - Hierarchical clustering
9. **Perceptron/SLP** - Single-layer neural network
10. **Multi-Layer Perceptron** - Deep neural network
11. **Custom Model** - Naive Bayes (outside classroom curriculum)

### Regression Models (4 from-scratch implementations):

1. **Linear Regression** - Basic relationship modeling
2. **Multiple Regression** - Multi-feature linear model
3. **Polynomial Regression** - Non-linear relationship capture
4. **XGBoost** - Gradient boosting ensemble

## 🔍 Feature Engineering Strategy

**Primary Transformations**:

- **Interaction Features**: Quantity × Unit Price = Total Transaction Value
- **Temporal Features**: Recency, frequency, monetary (RFM) analysis
- **Aggregation Features**: Customer-level statistics across transactions
- **Encoding**: Categorical variables (gender, category, location)
- **Scaling**: Numerical feature normalization

**Feature Selection**:

- Correlation analysis for multicollinearity
- Feature importance from tree-based models
- Domain knowledge-based selection

## 📈 Success Metrics & Evaluation

### Classification Metrics:

- **Primary**: F1-Score (harmonic mean of precision/recall)
- **Secondary**: Accuracy, Precision, Recall, AUC-ROC
- **Business**: Cost savings from retention campaigns

### Regression Metrics:

- **Primary**: RMSE (Root Mean Square Error)
- **Secondary**: R², MAE (Mean Absolute Error)
- **Business**: Revenue prediction accuracy

## 🚀 Business Impact & Applications

### Immediate Applications:

1. **Customer Retention**: Target at-risk customers with personalized offers
2. **Marketing Optimization**: Allocate budget to high-value customer segments
3. **Inventory Management**: Predict demand based on customer value predictions
4. **Pricing Strategy**: Dynamic pricing for different customer segments

### Long-term Benefits:

- **Revenue Growth**: 5-10% increase through better retention
- **Cost Reduction**: 15-20% decrease in customer acquisition costs
- **Customer Satisfaction**: Personalized experiences improve loyalty
- **Competitive Advantage**: Data-driven decision making

## 🔧 Technical Implementation

**Infrastructure**: MLOps pipeline with Airflow orchestration  
**Storage**: MinIO for data and model artifacts  
**Tracking**: MLflow for experiment management  
**Deployment**: Containerized microservices architecture

**Model Comparison**: Fair benchmarking with identical train/test splits, feature sets, and evaluation criteria between from-scratch and library implementations.

## 🎯 Expected Outcomess

1. **Production-ready models** for both regression and classification tasks
2. **Comprehensive comparison** between from-scratch and library implementations
3. **MLOps pipeline** for automated model training and deployment
4. **Business insights** for customer retention and value optimization
5. **Educational value** through deep understanding of ML algorithms

This project demonstrates the complete machine learning lifecycle from data ingestion to model deployment, emphasizing both theoretical understanding and practical implementation of machine learning algorithms in a real-world business context.
