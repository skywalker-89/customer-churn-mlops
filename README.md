# 🚀 Customer Churn Analysis & Value Estimation

## 👥 Role Assignments

- **Lead Engineer:** Infrastructure & Pipeline
- **Lead Data Scientist:** Feature Strategy & RL Agent (`src/lead_ds`)
- **Regression Engineer:** Value Estimation Model (`src/regression`)
- **Classification Engineer:** Churn Prediction Model (`src/classification`)

## 🛠️ How to Start (For Team)

1. **Clone the Repo:**

   ```bash
   git clone <REPO_URL>
   cd customer-churn-mlops
   ```

2. **Install Libraries:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Infrastructure (Docker):**

   ```bash
   docker-compose up -d
   ```

   - Airflow: http://localhost:8080 (User/Pass: airflow)
   - MinIO: http://localhost:9001 (User/Pass: minioadmin)

4. **Get the Data:**
   - I have processed the data into MinIO (`processed-data` bucket).
   - You can write your code in `src/` and read directly from MinIO or ask me for a sample file.

## ⚠️ Rules

- **DO NOT** upload data files (.csv, .parquet) to GitHub.
- **DO NOT** edit the `dags/` folder unless you are the Lead Engineer.
- **ALWAYS** inherit from `src/models_scratch/base.py` for your models.
