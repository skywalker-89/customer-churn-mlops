# 🚀 Project Setup Guide for ML Engineers

Welcome to the **Customer Churn MLOps** project! This guide is designed to get you from zero to a fully working environment. Follow every step carefully.

## 1. Prerequisites (Before you start)

Ensure you have the following installed on your machine:
*   **Docker Desktop**: Required to run the infrastructure. [Install here](https://www.docker.com/products/docker-desktop/).
    *   *Verify*: Run `docker --version` and `docker-compose --version` in your terminal.
*   **Git**: Required to clone the repository.

## 2. Project Installation

### 2.1 Clone the Repository
Open your terminal and run:
```bash
git clone <REPO_URL>
cd customer-churn-mlops
```

### 2.2 Download Raw Data
**Crucial Step**: The project relies on raw data files.
1.  **Download the Dataset**:
    *   Go to: [Maven Analytics Data Playground](https://mavenanalytics.io/data-playground/toy-store-e-commerce-database)
    *   Download the "Toy Store" dataset.
2.  **Place Data in Project**:
    *   Create the folder path if it doesn't exist: `data/raw/Toy_Store`
    *   Unzip/Move the downloaded CSV files (`orders.csv`, `website_sessions.csv`, etc.) into `data/raw/Toy_Store`.
3.  **Verify**:
    ```bash
    ls data/raw/Toy_Store
    ```
    > [!IMPORTANT]
    > You should see `.csv` files listed. If the directory is empty, the ingestion pipeline will fail.

## 3. Infrastructure Setup

We use Docker Compose to spin up all necessary services (Airflow, MinIO, MLflow, Postgres).

### 3.1 Start the Services
Run the following command in the project root:
```bash
docker-compose up -d
```
*   The `-d` flag runs it in "detached" mode (background).
*   **First time run**: This may take 5-10 minutes to download images and install dependencies.

### 3.2 Check Health
Run `docker-compose ps` to ensure all containers are `Up` (healthy).
You should see:
*   `airflow-webserver`
*   `airflow-scheduler`
*   `minio`
*   `mlflow_server`
*   `postgres` / `redis`

## 4. Accessing Services (Credentials)

Once the services are up, you can access them in your browser using these credentials:

| Service | URL | Username | Password | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Airflow** | [http://localhost:8080](http://localhost:8080) | `airflow` | `airflow` | Orchestrates the pipelines (DAGs). |
| **MinIO** | [http://localhost:9001](http://localhost:9001) | `minio_admin` | `minio_password` | Data Lake (S3 compatible). View raw & processed data. |
| **MLflow** | [http://localhost:5001](http://localhost:5001) | *None* | *None* | Tracks experiments and model metrics. |

---

## 5. The MLOps Workflow (Run this!)

Now that the infra is running, you need to execute the pipeline to prepare your environment for work.

### Step 1: Data Ingestion (Airflow)
This step converts the raw CSVs into Parquet format and uploads them to the MinIO Data Lake.

1.  Go to **Airflow** ([http://localhost:8080](http://localhost:8080)).
2.  Log in with `airflow` / `airflow`.
3.  Find the DAG named **`maven_ingestion_pipeline`**.
4.  **Unpause** it (toggle the switch to ON/Blue) if it's off.
5.  Click the **Trigger DAG** button (Release Play button ▶️) -> **Trigger DAG**.
6.  Click on the DAG name -> **Grid** view. Watch for the green success square.

> **Verify**: Go to **MinIO** ([http://localhost:9001](http://localhost:9001)). Login. Click `Object Browser` -> `raw-data` bucket. You should see `.parquet` files.

### Step 2: Validate Raw Data (Optional but Recommended)
Ensure the ingested data is clean.

1.  In Airflow, find **`data_quality_validation_standalone`**.
2.  Trigger the DAG.
3.  Wait for it to complete (Green).
4.  This checks for nulls, freshness, and schema validitiy.

### Step 3: Feature Engineering
This calculates features (like "time of day", "engagement depth") and prepares the training set.

1.  In Airflow, find **`feature_engineering_pipeline`**.
2.  Trigger the DAG.
3.  **Note**: This DAG will automatically trigger the `data_quality_validation` DAG upon completion.
4.  Wait for both to finish.

> **Verify**: Go to **MinIO**. Check the `processed-data` bucket. You should see `training_data.parquet`. This is the file you will use for training!

### Step 4: Model Training
Run the baseline training pipelines.

1.  In Airflow, find **`model_training_pipeline`**.
2.  Trigger the DAG.
3.  This will run:
    *   `train_classification_model`
    *   `train_regression_model`

> **Verify**: Go to **MLflow** ([http://localhost:5001](http://localhost:5001)). You should see new experiments listed with run metrics (Accuracy, RMSE, etc.).

---

## 6. How to Start Coding

You are now ready to work!

1.  **Code Location**:
    *   Write your Classification code in: `src/classification/`
    *   Write your Regression code in: `src/regression/`

2.  **Testing Your Code**:
    *   When you change your code, you don't need to restart Docker. The `src` folder is mounted.
    *   To test your changes, simply re-trigger the **`model_training_pipeline`** in Airflow.

3.  **Debugging**:
    *   **Airflow Logs**: Click a failed task box in Airflow -> **Logs** tab. This is where your print statements go.
    *   **Local Testing**: You can also run python scripts locally if you set the environment variables (ask Lead Engineer for help setting up local dev if needed, or stick to Airflow for consistency).

### 🆘 Common Issues
*   **"Connection Refused"**: Wait a minute. Services take time to start.
*   **"Bucket does not exist"**: Did you run Step 1 (Ingestion)?
*   **"Missing training_data.parquet"**: Did you run Step 3 (Feature Engineering)?
*   **"FATAL: database mlflow_db does not exist"**:
    *   This means the database wasn't created.
    *   Run `docker-compose down -v` to reset the volumes and start fresh.
    *   Then run `docker-compose up -d`.

Good luck! 🚀
