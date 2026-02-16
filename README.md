# Customer Churn MLOps Project

A complete MLOps pipeline for predicting customer churn using both regression (total sales prediction) and classification (churn prediction) models. Features automated data ingestion, feature engineering, model training with from-scratch implementations, and experiment tracking.

---

## 🎯 Project Overview

This project implements a machine learning operations (MLOps) pipeline for retail customer analytics:

- **Regression Models**: Predict customer lifetime value (`total_sales`)
- **Classification Models**: Predict customer churn (`churned` - whether customers will leave)
- **From-Scratch Implementations**: Custom ML algorithms built without sklearn
- **Production-Ready Infrastructure**: Airflow orchestration, MinIO storage, MLflow tracking

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Feature Eng.   │────▶│  Model Training │
│   (CSV files)   │     │  (Airflow DAG)  │     │  (Benchmarks)   │
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

---

## 🚀 Quick Start

### **1. Clone and Setup**

```bash
# Clone the repository
git clone https://github.com/skywalker-89/customer-churn-mlops.git
cd customer-churn-mlops

# Create required directories
mkdir -p logs dags plugins data
```

### **2. Start Infrastructure**

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# You should see all services running:
# - airflow-webserver
# - airflow-scheduler
# - airflow-worker
# - postgres
# - redis
# - minio
# - mlflow
```

### **3. Access Services**

Once all containers are running:

| Service | URL | Username | Password |
|---------|-----|----------|----------|
| **Airflow** | http://localhost:8080 | `airflow` | `airflow` |
| **MinIO** | http://localhost:9001 | `minio_admin` | `minio_password` |
| **MLflow** | http://localhost:5001 | - | - |

---

## 📊 Running the Data Pipeline

### **Step 1: Data Ingestion (Run Once)**

Loads raw CSV files from `data/Toy_Store/` into MinIO.

**Via Airflow UI:**
1. Go to http://localhost:8080
2. Login with `airflow` / `airflow`
3. Find DAG: **`retail_data_ingestion_pipeline`**
4. Click the **▶️ Play** button (Trigger DAG)
5. Wait for completion (should turn green)

**Via Command Line:**
```bash
# Trigger the DAG
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags trigger retail_data_ingestion_pipeline

# Check status
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list-runs -d retail_data_ingestion_pipeline
```

**What it does:**
- Reads all CSV files from `data/Toy_Store/`
- Uploads to MinIO bucket: `raw-data`
- Validates data integrity

---

### **Step 2: Feature Engineering (Run Once)**

Processes raw data and creates ML-ready features.

**Via Airflow UI:**
1. Go to http://localhost:8080
2. Find DAG: **`retail_feature_engineering_pipeline`**
3. Click the **▶️ Play** button (Trigger DAG)
4. Wait for completion (should turn green)

**Via Command Line:**
```bash
# Trigger the DAG
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags trigger retail_feature_engineering_pipeline

# Check status
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list-runs -d retail_feature_engineering_pipeline
```

**What it does:**
- Loads raw data from MinIO
- Creates engineered features:
  - Interaction features (`quantity * unit_price`)
  - Aggregations (total purchases, avg order value)
  - Recency/frequency metrics
  - Category encodings
- Creates target variables:
  - `total_sales` (regression target)
  - `churned` (classification target)
- Saves to MinIO bucket: `processed-data/training_data.parquet`

---

### **Step 3: Model Training (Automated Hourly)**

The model training DAG runs automatically every hour or can be triggered manually.

**Via Airflow UI:**
1. Find DAG: **`model_training_pipeline`**
2. Click **▶️ Play** to trigger manually

**What it does:**
- Validates training data exists
- Runs regression benchmark (your work):
  - Linear Regression (from scratch)
  - Multiple Regression (from scratch)
  - Polynomial Regression (from scratch)
  - XGBoost (from scratch)
  - XGBoost (sklearn) for comparison
- Runs classification benchmark (your friend's work):
  - 11 from-scratch classification models
  - 1 sklearn model for comparison
- Saves all models to MinIO bucket: `models`
- Logs metrics to MLflow

---

## 📁 Project Structure

```
customer-churn-mlops/
├── dags/                              # Airflow DAGs
│   ├── retail_ingestion_dag.py        # Data ingestion
│   ├── retail_feature_engineering_dag.py  # Feature engineering
│   ├── model_training_dag.py          # Model training orchestration
│   └── data_quality_dag.py            # Data validation
│
├── src/                               # Source code
│   ├── models_scratch/                # From-scratch ML implementations
│   │   ├── base.py                    # Base model class
│   │   ├── linear_regression.py       # Linear regression
│   │   ├── polynomial_regression.py   # Polynomial regression
│   │   ├── xgboost.py                 # XGBoost from scratch
│   │   └── ...                        # More models
│   │
│   ├── regression/
│   │   └── retail_regression_benchmark.py  # Regression model benchmark
│   │
│   └── classification/
│       └── retail_classification_benchmark.py  # Classification model benchmark
│
├── data/                              # Raw data directory
│   └── Toy_Store/                     # Raw CSV files
│
├── docker-compose.yaml                # Infrastructure definition
├── requirements.txt                   # Python dependencies
├── CLASSIFICATION_HANDOFF.md          # Instructions for classification team
└── README.md                          # This file
```

---

## 🔧 Configuration

### **Environment Variables**

Set these in `.env` file (optional):

```bash
AIRFLOW_UID=50000
AIRFLOW_IMAGE_NAME=apache/airflow:2.10.4

# MinIO credentials
MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=minio_password

# Airflow credentials
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
```

### **MinIO Buckets**

The DAGs automatically create these buckets:
- `raw-data` - Raw CSV files
- `processed-data` - Feature-engineered data
- `models` - Trained model artifacts
- `mlflow` - MLflow artifacts

---

## 🧪 Monitoring and Debugging

### **Check DAG Status**

```bash
# List all DAGs
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list

# View last 5 runs of a DAG
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list-runs -d retail_data_ingestion_pipeline --limit 5
```

### **View Logs**

```bash
# Airflow scheduler logs
docker logs customer-churn-mlops-airflow-scheduler-1 -f

# Airflow worker logs
docker logs customer-churn-mlops-airflow-worker-1 -f

# Specific task logs (via Airflow UI)
# Go to http://localhost:8080 → DAGs → Click DAG → Click Task → View Logs
```

### **Check MinIO Data**

1. Go to http://localhost:9001
2. Login: `minio_admin` / `minio_password`
3. Browse buckets:
   - `raw-data` → See uploaded CSV files
   - `processed-data` → See `training_data.parquet`
   - `models` → See trained model `.pkl` files

### **View MLflow Experiments**

1. Go to http://localhost:5001
2. Browse experiments:
   - `retail_regression_benchmark` - Regression model metrics
   - `retail_classification_benchmark` - Classification model metrics
3. Compare model performance, view metrics, download artifacts

---

## 🛠️ Troubleshooting

### **Services Not Starting**

```bash
# Check service status
docker-compose ps

# Restart all services
docker-compose restart

# Full reset (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### **DAG Not Appearing in Airflow**

```bash
# Refresh DAGs
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  airflow dags list-import-errors

# Check syntax errors
docker exec -it customer-churn-mlops-airflow-scheduler-1 \
  python /opt/airflow/dags/retail_ingestion_dag.py
```

### **Out of Memory**

If containers crash due to memory:
1. Increase Docker Desktop memory limit (8GB+ recommended)
2. Reduce dataset size in feature engineering DAG
3. Reduce number of parallel workers

### **Permission Errors**

```bash
# Fix Airflow permissions
sudo chown -R 50000:0 logs/ dags/ plugins/

# Or set AIRFLOW_UID
echo "AIRFLOW_UID=$(id -u)" > .env
docker-compose down
docker-compose up -d
```

---

## 📚 Additional Resources

### **For Classification Team**
- Read: `CLASSIFICATION_HANDOFF.md`
- Implement 11 from-scratch classification models
- Run benchmark and send trained models

### **Key Concepts**
- **Churn**: When a customer stops doing business with the company
- **From-Scratch Models**: ML algorithms implemented without sklearn (use only NumPy)
- **Warm Start**: Continue training from existing model (incremental learning)

---

## 🤝 Contributing

### **Team Structure**
- **Lead Engineer (Regression)**: Implements 4 from-scratch regression models
- **ML Engineer (Classification)**: Implements 11 from-scratch classification models

### **Development Workflow**
1. Build models in `src/models_scratch/`
2. Inherit from `BaseModel` class
3. Test locally: `python -m src.regression.retail_regression_benchmark`
4. Run via Airflow DAG
5. Download trained models from MinIO
6. Share models with team

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎓 University Project

This project is part of a machine learning course assignment demonstrating:
- End-to-end MLOps pipeline
- From-scratch ML algorithm implementations
- Fair comparison between custom and library models
- Production-ready infrastructure
- Automated workflow orchestration

**Built with ❤️ for learning and experimentation**
