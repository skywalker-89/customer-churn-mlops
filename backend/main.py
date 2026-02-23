from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from routers import metrics, predict

app = FastAPI(title="Customer Churn & Revenue Dashboard API")

# Mount data directory for visualizations
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
if os.path.exists(DATA_PATH):
    app.mount("/static/data", StaticFiles(directory=DATA_PATH), name="data")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(metrics.router, prefix="/api", tags=["metrics"])
app.include_router(predict.router, prefix="/api", tags=["predictions"])


@app.get("/")
def root():
    return {"message": "MLOps Dashboard API is running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
