# Customer Churn MLOps Project

A full MLOps pipeline for predicting customer churn and total sales on a large-scale retail dataset (~1M rows, 47 features). Features automated data ingestion, feature engineering, from-scratch ML algorithm implementations, and experiment tracking.

---

## 🎯 Project Overview

This project implements a machine learning operations (MLOps) pipeline for retail customer analytics:

- **Regression Models**: Predict `total_sales` (customer purchase amount)
- **Classification Models**: Predict `churned` (whether a customer will churn)
- **From-Scratch Implementations**: Custom ML algorithms built using only NumPy — no sklearn for training
- **Production-Ready Infrastructure**: Airflow orchestration, MinIO storage, MLflow tracking

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Feature Eng.   │────▶│  Model Training │
│ retail_data.csv │     │  (Airflow DAG)  │     │  (Benchmarks)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
    ┌─────────┐            ┌─────────┐            ┌──────────┐
    │  MinIO  │            │  MinIO  │            │  MLflow  │
    │ (Raw)   │            │(Processed)           │(Tracking)│
    └─────────┘            └─────────┘            └──────────┘
```

**Components:**
- **Airflow**: Workflow orchestration
- **MinIO**: S3-compatible object storage for data and models
- **MLflow**: Experiment tracking and model registry
- **PostgreSQL**: Metadata storage (Airflow + MLflow)
- **Docker Compose**: Infrastructure management

---

## 📋 Prerequisites

- **Docker** & **Docker Compose** installed
- **8GB+ RAM** recommended
- **10GB+ disk space**
- `data/raw/retail_data.csv` placed in the project root (not included in repo due to size)

---

## 🚀 Quick Start

### **1. Clone and Setup**

```bash
git clone https://github.com/skywalker-89/customer-churn-mlops.git
cd customer-churn-mlops

# Create required directories
mkdir -p logs dags plugins data/raw
```

### **2. Add the Dataset**

Place `retail_data.csv` inside `data/raw/`:

```
data/
└── raw/
    └── retail_data.csv   ← ~1M rows, 47 columns
```

### **3. Start Infrastructure**

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

You should see all services running:
- `airflow-webserver`, `airflow-scheduler`, `airflow-worker`
- `postgres`, `redis`
- `minio`, `mlflow`

### **4. Access Services**

| Service | URL | Username | Password |
|---------|-----|----------|----------|
| **Airflow** | http://localhost:8080 | `airflow` | `airflow` |
| **MinIO** | http://localhost:9001 | `minio_admin` | `minio_password` |
| **MLflow** | http://localhost:5001 | — | — |

---

## 📊 Running the Pipeline

### **Step 1: Data Ingestion**

Reads `data/raw/retail_data.csv` and uploads it to MinIO as a Parquet file.

**DAG:** `retail_data_ingestion_pipeline`

**Via Airflow UI:**
1. Go to http://localhost:8080 → Login with `airflow` / `airflow`
2. Find DAG: **`retail_data_ingestion_pipeline`**
3. Click **▶️ Trigger DAG**

**Via CLI:**
```bash
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags trigger retail_data_ingestion_pipeline
```

**What it does:**
- Reads `retail_data.csv` (~1M rows)
- Converts to Parquet and uploads to MinIO bucket: `raw-data`

---

### **Step 2: Feature Engineering**

Processes raw data and produces the final ML-ready training dataset.

**DAG:** `retail_feature_engineering_pipeline`

**Via Airflow UI:**
1. Find DAG: **`retail_feature_engineering_pipeline`**
2. Click **▶️ Trigger DAG**

**Via CLI:**
```bash
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags trigger retail_feature_engineering_pipeline
```

**What it does:**
- Loads `retail_data.parquet` from MinIO
- Drops ID columns (`customer_id`, `transaction_id`, etc.)
- One-hot encodes 27 categorical variables
- Creates derived features:
  - `quantity_times_price` — interaction feature
  - `engagement_score` — app + social media usage
  - `recency_ratio` — days since last purchase / 365
  - `online_preference` — online vs in-store purchase ratio
- Synthesizes targets:
  - `total_sales` = `quantity × unit_price × (1 − discount) + noise`
  - `churned` = derived from recency and purchase frequency
- Saves to MinIO: `processed-data/training_data.parquet`
- Auto-triggers the **Data Quality DAG** on completion

---

### **Step 3: Model Training**

Runs regression and classification benchmarks in parallel.

**DAG:** `model_training_pipeline` (runs weekly or trigger manually)

**Via Airflow UI:**
1. Find DAG: **`model_training_pipeline`**
2. Click **▶️ Trigger DAG**

**What it does:**
1. Validates `training_data.parquet` exists in MinIO
2. **Regression benchmark** — trains 4 from-scratch models + 1 sklearn baseline:
   - Linear Regression (gradient descent)
   - Multiple Regression (gradient descent)
   - Polynomial Regression (degree 2, mini-batch SGD)
   - XGBoost (from scratch, gradient boosting)
   - XGBoost (sklearn, for comparison)
3. **Classification benchmark** — trains 11 from-scratch models + 1 sklearn baseline:
   - Logistic Regression, Decision Tree, Random Forest
   - SVM, Random Forest + PCA, SVM + PCA
   - K-Means Clustering, Agglomerative Clustering
   - Perceptron, MLP, Custom Model
   - Random Forest Classifier (sklearn, for comparison)
4. Saves all trained models to MinIO bucket: `models`
5. Logs all metrics to MLflow

---

## 📁 Project Structure

```
customer-churn-mlops/
├── dags/                                    # Airflow DAGs
│   ├── retail_ingestion_dag.py              # Data ingestion
│   ├── retail_feature_engineering_dag.py    # Feature engineering
│   ├── model_training_dag.py                # Training orchestration
│   ├── model_evaluation_dag.py              # Model evaluation & comparison
│   └── data_quality_dag.py                  # Data validation
│
├── src/                                     # Source code
│   ├── models_scratch/                      # From-scratch ML implementations
│   │   ├── base.py                          # BaseModel class
│   │   ├── linear_regression.py
│   │   ├── multiple_regression.py
│   │   ├── polynomial_regression.py
│   │   ├── xgboost.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── random_forest_pca.py
│   │   ├── svm_pca.py
│   │   ├── pca.py
│   │   ├── kmeans_clustering.py
│   │   ├── agglomerative_clustering.py
│   │   ├── perceptron.py
│   │   ├── mlp.py
│   │   └── custom_model.py
│   │
│   ├── regression/
│   │   └── retail_regression_benchmark.py   # Regression benchmark runner
│   │
│   └── classification/
│       └── retail_classification_benchmark.py  # Classification benchmark runner
│
├── data/
│   └── raw/
│       └── retail_data.csv                  # ~1M rows, 47 features (not in repo)
│
├── docker-compose.yaml                      # Infrastructure definition
├── requirements.txt                         # Python dependencies
├── CLASSIFICATION_HANDOFF.md                # Instructions for classification team
└── README.md                                # This file
```

---

## 🔧 Configuration

### **Environment Variables**

Set in `.env` (optional, defaults shown):

```bash
AIRFLOW_UID=50000
AIRFLOW_IMAGE_NAME=apache/airflow:2.10.4

MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=minio_password

_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
```

### **MinIO Buckets**

DAGs auto-create these buckets:
- `raw-data` — Raw Parquet from CSV ingestion
- `processed-data` — Feature-engineered training data
- `models` — Trained model `.pkl` files
- `mlflow` — MLflow artifacts

---

## 🧪 Monitoring & Debugging

### **Check DAG Status**

```bash
# List all DAGs
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list

# View recent runs
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list-runs -d retail_data_ingestion_pipeline --limit 5
```

### **View Logs**

```bash
# Scheduler logs
docker logs customer-churn-mlops-airflow-scheduler-1 -f

# Worker logs
docker logs customer-churn-mlops-airflow-worker-1 -f

# Per-task logs: Airflow UI → DAGs → Click DAG → Click Task → View Logs
```

### **Check MinIO**

1. Go to http://localhost:9001 → Login: `minio_admin` / `minio_password`
2. Browse:
   - `raw-data` → `retail_data.parquet`
   - `processed-data` → `training_data.parquet`
   - `models` → trained `.pkl` files

### **View MLflow Experiments**

1. Go to http://localhost:5001
2. Experiments:
   - `retail_regression_benchmark` — RMSE, MAE, R², MAPE per model
   - `retail_classification_benchmark` — Accuracy, Precision, Recall, F1 per model

---

## 🛠️ Troubleshooting

### **Services Not Starting**

```bash
docker-compose ps
docker-compose restart

# Full reset (WARNING: deletes all data)
docker-compose down -v && docker-compose up -d
```

### **DAG Not Appearing in Airflow**

```bash
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list-import-errors
```

### **Out of Memory**

- Increase Docker Desktop memory to 8GB+
- Reduce `sample_size` in the benchmark scripts' `load_retail_data()` call

### **Permission Errors**

```bash
# Fix Airflow file permissions
sudo chown -R 50000:0 logs/ dags/ plugins/

# Or set AIRFLOW_UID to your user
echo "AIRFLOW_UID=$(id -u)" > .env
docker-compose down && docker-compose up -d
```

---

## 🤝 Team Structure

| Role | Responsibility |
|------|---------------|
| **Lead Engineer (Regression)** | 4 from-scratch regression models + pipeline orchestration |
| **ML Engineer (Classification)** | 11 from-scratch classification models |

### **Development Workflow**

1. Implement your model in `src/models_scratch/` inheriting from `BaseModel`
2. Register it in the corresponding benchmark script
3. Test locally: `python -m src.regression.retail_regression_benchmark`
4. Run via the Airflow DAG — models auto-save to and load from MinIO (warm start)

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🎓 University Project

This project is part of a machine learning course demonstrating:
- End-to-end MLOps pipeline on a real-scale retail dataset
- From-scratch ML algorithm implementations (NumPy only)
- Fair benchmarking between custom and library models
- Production-ready infrastructure with Docker, Airflow, MinIO, and MLflow

**Built with ❤️ for learning and experimentation**
