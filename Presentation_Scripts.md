# 🎤 Presentation Speaker Scripts
## Customer Churn Prediction & Analysis — [01276343] ML Project
### March 9, 2026 | ~18–20 minutes total

---

> **Split Overview**
>
> | Member | Slides | Focus Area | ~Time |
> |--------|--------|-----------|-------|
> | **Member 1** | 1, 2, 3, 4 | Intro, Problem, Dataset | 4–5 min |
> | **Member 2** | 5, 6, 7, 8, 9 | Features, Preprocessing, Regression | 5–6 min |
> | **Member 3** | 10, 11, 12, 13 | All Classification Models + Novel | 5–6 min |
> | **Member 4** | 14, 15, 16, 17, 18 | Benchmarking, Results, Conclusion | 4–5 min |

---

---

## 👤 MEMBER 1 — Slides 1–4
### *Introduction · Problem Statement · Dataset*

---

### 🔲 SLIDE 1 — Title Slide

> *(Stand at the front, let the slide settle for 2 seconds before speaking)*

"Good morning / Good afternoon, everyone. My name is [Name], and on behalf of our team I'd like to welcome you to our final ML project presentation.

Our project is titled **'Customer Churn Prediction and Analysis'** — a dual-task machine learning problem where we predict both customer lifetime value and whether a customer will stop buying from a retail business.

We worked on this as a complete, end-to-end MLOps pipeline — meaning not just models, but automated training, experiment tracking, and model storage — which I'll walk you through.

Let me hand it over to my teammate in a moment, but first let me give you a quick road-map of what we'll cover today."

---

### 🔲 SLIDE 2 — Agenda

"On the left you can see the first half of our presentation — we'll cover the problem, the dataset, our feature engineering, then walk through every model we implemented.

On the right, we move into our novel models, a full benchmarking comparison, ROC analysis, and finally our conclusions.

In total we trained and evaluated **17 machine-learning models** from scratch — 5 regression models and 12 classification models — and we'll compare them all side-by-side.

Let's begin."

---

### 🔲 SLIDE 3 — Problem Statement

"So — what exactly is customer churn?

Churn happens when a customer simply stops buying from you — they go quiet, switch to a competitor, or disengage entirely. In retail, the average annual churn rate sits at about **25 to 30 percent**, which is enormous.

Why does this matter financially? Acquiring a new customer costs **5 to 25 times more** than retaining an existing one. A 5% improvement in retention can translate to a **revenue lift of 5 to 10 percent**.

Now — why can't we just use simple business rules to catch churners early? The answer is on the right card: our dataset has **over 70 features**. The interactions between customer behavior, transaction history, product ratings, and promotional channels are simply too complex for manual rules.

That's exactly where Machine Learning comes in — and specifically, why we implemented both a **regression task** to predict customer lifetime value, and a **classification task** to flag customers at risk of churning."

---

### 🔲 SLIDE 4 — Dataset Overview

"Our dataset is a synthetic retail customer analytics dataset — it contains **1,000 customer records** with **more than 70 features** spanning six categories: demographics, behavior, transaction history, product analytics, marketing, and customer engagement.

We defined **two target variables**:
- `total_sales` — a continuous value representing how much a customer has spent in total. This is our **regression target**.
- `churned` — a binary label, 0 for retained and 1 for churned. This is our **classification target**.

The class balance is roughly **70% retained, 30% churned**, so there's a mild class imbalance that we accounted for in our evaluation by prioritising F1-score over raw accuracy.

On the right you can see our full feature correlation heatmap — it gives a sense of the feature density we were working with, and the clusters helped us identify groups of correlated variables.

With that, I'll hand over to [Member 2] who will walk you through how we engineered features and built our models."

---

---

## 👤 MEMBER 2 — Slides 5–9
### *Feature Engineering · Preprocessing · Regression Track*

---

### 🔲 SLIDE 5 — Feature Analysis & Storytelling

> *(Member 2 takes the floor)*

"Thank you, [Member 1].

Before we trained any model, we spent time understanding *which* features actually matter and *why*. This is what we call feature storytelling.

The single most impactful engineered feature was a simple multiplication: **quantity times unit price** — giving us the total transaction value per purchase. This interaction term turned out to be the strongest predictor of `total_sales`.

For churn specifically, **recency** — meaning how many days since a customer's last purchase — was the clearest early warning signal. If a loyal customer suddenly goes 60 days without a purchase, that's a red flag.

**Loyalty membership years** correlated negatively with churn — the longer someone has been a member, the less likely they are to leave.

And on the right you can see our feature relationship chart, which visualises the pairwise dependencies between our key numerical features.

We also ran a full correlation analysis — visible in later slides — to detect multicollinearity and guide feature selection."

---

### 🔲 SLIDE 6 — Data Preprocessing Pipeline

"Here is our full preprocessing pipeline — 7 steps, applied consistently to produce a clean, model-ready dataset.

Step 1: **Null handling** — we used median imputation for numerical columns and mode for categorical ones, since our dataset had some missingness in transaction-level fields.

Step 2: **Encoding** — all nominal categorical variables like gender, product category, and store location were one-hot encoded.

Step 3: **Feature engineering** — we added the quantity-times-unit-price interaction term directly at this stage.

Step 4: **Aggregation** — we computed per-customer RFM statistics: recency, frequency, and average order value, collapsing transaction-level data to customer-level.

Step 5: **Scaling** — all numerical features were standardised using Z-score scaling so gradient-descent models converge efficiently.

Step 6: **Train/test split** — we used an **80/20 stratified split** that was applied **identically** to every single model. Fair benchmarking was critical to us.

Step 7: **Storage** — the final processed dataset was saved to our MinIO object store as a Parquet file for efficient retrieval by our Airflow training DAG."

---

### 🔲 SLIDE 7 — Methodology Overview

"Our methodology is split into two parallel tracks.

On the **left — the Regression Track** — we predict `total_sales`, a continuous value. We implemented Linear, Multiple, Polynomial, and XGBoost regression — all built from scratch. We also ran XGBoost through sklearn purely as a sanity check to validate our implementation.

On the **right — the Classification Track** — we predict `churned`, a binary label. Nine distinct from-scratch classifiers plus two variants using PCA dimensionality reduction, and one sklearn baseline.

Every single model on both tracks was implemented using **NumPy only** — no sklearn training code. Library implementations were used exclusively for result validation.

Let me walk you through the regression track first."

---

### 🔲 SLIDE 8 — Regression Models

"We implemented four regression models from scratch.

**Linear Regression** — the simplest baseline. We optimise weights using gradient descent on a single feature. It gives us a clean understanding of the baseline linear relationship.

**Multiple Regression** — extends this to all features simultaneously, learning a weight vector across the full feature space.

**Polynomial Regression** — we expand to degree-2 features to capture non-linear relationships; regularisation prevents overfitting on the expanded feature set.

And then our **novel model — XGBoost** — which is a gradient-boosted decision tree ensemble. Each new tree learns the residual errors of the previous trees. We implemented the full boosting loop, tree splitting logic, and shrinkage from scratch. This is the model that dramatically outperforms all others — and I'll show you exactly how much in the next slide."

---

### 🔲 SLIDE 9 — Regression Results

"Here are the numbers — and they really tell a story.

Looking at the R² column: Linear Regression achieves **0.41**, Multiple Regression barely reaches **0.10** — it actually underfit because adding more features without transformation didn't help. Polynomial gets us to **0.74**.

And then XGBoost scratch: **R² of 0.9946** and an RMSE of just **119.9** — compared to 1,248 for Linear Regression. That's more than a **10× improvement in RMSE**.

The sklearn XGBoost validation gives R² of 0.9989 and RMSE 54.8 — so our from-scratch implementation is very close to the library result, which validates our implementation.

The two plots at the bottom show our regression metrics comparison and the predicted-vs-actual plot for XGBoost — notice how tightly the points cluster around the diagonal. That's near-perfect prediction.

I'll now pass to [Member 3] who will take us through our classification models."

---

---

## 👤 MEMBER 3 — Slides 10–13
### *Classification Models · Novel Classifier*

---

### 🔲 SLIDE 10 — Classification Part 1: LogReg · DTree · RF

> *(Member 3 takes the floor)*

"Thank you, [Member 2].

I'll now walk you through our classification models — we have 11 from scratch, so I'll group them into three slides starting with the foundational ones.

**Logistic Regression** — our baseline binary classifier. We use the sigmoid activation and binary cross-entropy loss optimised with gradient descent. It achieved an **accuracy of 83.3%** and an **F1 score of 0.675** — which will actually hold up as one of our best all-around results.

**Decision Tree** — we implemented recursive binary splitting using Gini impurity. Accuracy of **84.1%** — the highest raw accuracy in Part 1 — but F1 is slightly lower at 0.639 because it's more biased toward the majority class.

**Random Forest** — 100 decision trees, majority vote. Accuracy **84.3%**, F1 **0.645**. Nearly identical to our sklearn Random Forest baseline — which validates the implementation perfectly.

On the right you can see the confusion matrices for these three models. Notice that the Random Forest and Decision Tree have very similar patterns — high true negatives but some missed churners."

---

### 🔲 SLIDE 11 — Classification Part 2: SVM · K-Means · Agglomerative

"Next group.

**Support Vector Machine** — we implemented an SVM from scratch using gradient-based optimisation of the hinge loss. Accuracy is **75.7%** — lower than Random Forest — but look at the **recall: 0.903**. That means it correctly identifies over 90% of actual churners. For a business use-case where missing a churner is costly, high recall is very valuable.

We also tested **SVM with PCA** — applying dimensionality reduction first. The results were essentially identical, confirming that PCA successfully compressed the feature space without losing predictive information.

**K-Means Clustering** — this is an unsupervised model, so we adapted it by mapping the 2 clusters to labels based on majority class. Its F1 of **0.241** is low — which is expected, because it has no access to labels during training. This is more of a customer segmentation tool than a churn predictor.

**Agglomerative Clustering** — hierarchical clustering with ward linkage. This one surprised us — it achieved an accuracy of **82.5%** and F1 of **0.675**, competitive with our best supervised models despite being unsupervised.

You can see all three confusion matrices on the right side of this slide."

---

### 🔲 SLIDE 12 — Classification Part 3: Perceptron · MLP · Custom Model

"The final group of classifiers.

**Perceptron / SLP** — the simplest neural model: single layer, step activation function, threshold-based weight update. Accuracy of **78.2%**, F1 **0.533**. As expected, it's limited by its linear decision boundary.

**Multi-Layer Perceptron** — two hidden layers with ReLU activation, sigmoid output layer, trained with an Adam-like gradient descent update rule. Accuracy **82.7%**, F1 **0.673**. A solid neural network result fully implemented in NumPy.

And our **novel classifier — Naive Bayes**, which is outside the classroom curriculum. We implemented Gaussian Naive Bayes from scratch: computing the mean and standard deviation per class per feature, then applying Bayes' theorem at inference. It achieved **accuracy 82.5%** and **F1 0.675** — matching Logistic Regression exactly and outperforming Perceptron and many other models despite being conceptually simpler.

What makes Naive Bayes special here is that it requires no iterative training — inference is direct probability computation, making it extremely fast in production."

---

### 🔲 SLIDE 13 — Novel / New Models

"Let me bring both novel models together for a direct comparison.

On the left — **XGBoost for regression**. The jump from Polynomial Regression R²=0.74 to XGBoost R²=0.9946 is enormous. RMSE drops from 825 to 119. The sklearn validation at R²=0.9989 is very close, confirming the implementation is sound. This is unambiguously the best regression model.

On the right — **Naive Bayes for classification**. With F1=0.675 it matches the top classification models and outperforms more complex models like MLP (0.673) and Decision Tree (0.639). It also requires zero training iterations — computationally it's the cheapest classifier we have.

Both novel models were implemented fully from scratch, and both deliver strong results that justify going beyond the standard curriculum.

I'll now hand over to [Member 4] for the full benchmarking comparison, ROC analysis, and our conclusions."

---

---

## 👤 MEMBER 4 — Slides 14–18
### *Benchmarking · ROC · Discussion · Conclusion*

---

### 🔲 SLIDE 14 — Benchmarking Table

> *(Member 4 takes the floor)*

"Thank you, [Member 3].

Now let's look at the complete picture — all 17 models side by side.

Looking at the **classification table** on the left: the top accuracy cluster sits around **84%** — Random Forest, Random Forest + PCA, and the sklearn RF all reach 0.843. However, **accuracy alone is misleading** with imbalanced classes. When we look at F1, the leaders are Logistic Regression, Agglomerative Clustering, and our novel Naive Bayes — all tied at **0.675**.

One standout pattern: **SVM has the highest recall at 0.903** — meaning it catches 90% of all churners — at the cost of lower precision. Depending on the business context, that trade-off might be exactly what you want.

**K-Means is the weakest** at F1=0.241 — but that's by design; clustering isn't meant for direct supervised prediction.

On the regression side — the story is simple: **XGBoost wins by a large margin**, with R²=0.9946 versus 0.74 for the next best model. The sklearn XGBoost at 0.9989 confirms our implementation quality.

The chart on the right gives a visual representation of all classification metrics together."

---

### 🔲 SLIDE 15 — ROC & AUC

"ROC curves show how a model balances true positive rate against false positive rate across different decision thresholds. The AUC — Area Under the Curve — summarises this into a single number: 1.0 is perfect, 0.5 is random.

Based on our results:
- **Random Forest** — highest accuracy, likely highest AUC
- **Logistic Regression & Naive Bayes** — best F1, strong AUC
- **SVM** — high recall favors sensitivity, potentially very high AUC
- **K-Means** — AUC near 0.5, as expected for an unsupervised method mapped to labels

Our full ROC curves are tracked inside our MLflow experiment `retail_classification_benchmark`. If you'd like we can show the MLflow dashboard live after the presentation.

The classification metrics chart on the right gives a model-by-model visual summary across accuracy, precision, recall, and F1 simultaneously — which is useful for picking the right model for a given business objective."

---

### 🔲 SLIDE 16 — Discussion & Key Findings

"Let me now step back and discuss what we learned.

**Which models won?** In regression: XGBoost — unambiguously. In classification: it depends on your objective. If you want overall accuracy, choose Random Forest. If you want to catch the most churners, use SVM. If you want a balanced F1 with fast inference, use Logistic Regression or Naive Bayes.

**From-scratch vs sklearn validation** — this was one of our key requirements. Our XGBoost scratch achieved R²=0.9946 vs sklearn's 0.9989 — a gap of less than 0.5%. Our Random Forest scratch vs sklearn was essentially identical. This confirms that our implementations are correct.

**Overfitting and underfitting observations** — Multiple Regression underfit with only R²=0.10 because adding raw features without the interaction term doesn't improve a linear model much. K-Means is fundamentally the wrong tool for a supervised classification task — it's designed for segmentation, not prediction.

**Feature insight** — the interaction feature of quantity × unit price was the single biggest contributor to regression performance. Recency and loyalty membership years were the top churn signals."

---

### 🔲 SLIDE 17 — Conclusion

"To summarise what we achieved:

We implemented **17 machine learning models** across two tasks — all built from scratch in NumPy. Our novel models — XGBoost for regression and Naive Bayes for classification — both demonstrated stronger or competitive results versus the baseline curriculum models.

We built a **complete MLOps pipeline**: Airflow orchestrates the workflow, MinIO stores all data and model artifacts, and MLflow tracks every experiment with full metrics logging. Everything runs containerised via Docker Compose.

Our from-scratch implementations were rigorously validated against sklearn equivalents in every case.

On the limitations side: the dataset is synthetic, so real-world performance may differ. K-Means is not well-suited to supervised prediction. Our MLP is constrained by the from-scratch NumPy requirement — a PyTorch version would likely perform better and train faster.

For future work, we'd love to apply LSTM or Transformer architectures to model sequential transaction patterns, integrate live churn alerting, and add AutoML hyperparameter tuning into the Airflow DAG.

Thank you."

---

### 🔲 SLIDE 18 — References

> *(Hold for questions)*

"Our references are listed here — from the foundational papers for Random Forest, XGBoost, and SVM, to the libraries and infrastructure tools we used: Scikit-learn, Apache Airflow, MLflow, and MinIO.

The dataset is a synthetic retail customer analytics dataset of our own construction.

We're happy to take any questions — whether on the model implementations, the pipeline architecture, or the results.

*(Pause)*

Thank you very much for your time."

---

---

## ⏱️ Timing Guide

| Member | Slides | Target Time |
|--------|--------|-------------|
| Member 1 | 1–4 | ~4 min |
| Member 2 | 5–9 | ~5 min |
| Member 3 | 10–13 | ~5 min |
| Member 4 | 14–18 | ~4–5 min |
| **Total** | | **~18–19 min** |

---

## 💡 Presentation Tips

- Each member should **stand up, face the audience**, and not read directly from the screen.
- The **handover lines** are already built in — use the "I'll now hand over to [Name]" phrases as natural transitions.
- Keep bullet points on screen as **visual anchors** — elaborate with your words, don't just read the slide.
- Practice your slide **at least twice** so you're comfortable with the timing.
- If the professor asks about a specific model, the responsible member should step up to answer.
