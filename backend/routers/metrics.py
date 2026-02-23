import json
import glob
import os
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

MLRUNS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../mlruns")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
VISUALIZATION_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../visualization_output"
)


def get_latest_visualization_paths():
    """
    Parse the latest summary file to get visualization paths.
    """
    summary_files = glob.glob(os.path.join(DATA_PATH, "summary_*.txt"))
    if not summary_files:
        return {}

    # Get latest file
    latest_summary = max(summary_files, key=os.path.getctime)

    paths = {"classification": None, "regression": None}

    try:
        with open(latest_summary, "r") as f:
            content = f.read()

        # Parse classification path
        if "CLASSIFICATION VISUALIZATIONS:" in content:
            class_section = content.split("CLASSIFICATION VISUALIZATIONS:")[1].split(
                "REGRESSION VISUALIZATIONS:"
            )[0]
            for line in class_section.split("\n"):
                if "Location:" in line:
                    # Extract path: visualization_output/classification_...
                    rel_path = line.split("Location:")[1].strip()
                    # We need the part after visualization_output/ or just the folder name
                    if "visualization_output/" in rel_path:
                        folder_name = rel_path.split("visualization_output/")[1]
                    else:
                        folder_name = rel_path

                    # Construct URL path: /static/data/ + folder name + /metrics_comparison.png
                    paths["classification"] = (
                        f"/static/data/{folder_name}/metrics_comparison.png"
                    )
                    break

        # Parse regression path
        if "REGRESSION VISUALIZATIONS:" in content:
            reg_section = content.split("REGRESSION VISUALIZATIONS:")[1]
            for line in reg_section.split("\n"):
                if "Location:" in line:
                    rel_path = line.split("Location:")[1].strip()
                    if "visualization_output/" in rel_path:
                        folder_name = rel_path.split("visualization_output/")[1]
                    else:
                        folder_name = rel_path

                    paths["regression"] = (
                        f"/static/data/{folder_name}/metrics_comparison.png"
                    )
                    break

    except Exception as e:
        print(f"Error parsing summary file: {e}")

    return paths


@router.get("/metrics")
async def get_metrics():
    """
    Load metrics from the latest evaluation report JSON file.
    Also returns paths to static visualization files.
    """
    metrics_data = {
        "classification": [],
        "regression": [],
        "visualizations": get_latest_visualization_paths(),
    }

    # Find evaluation report files
    # We prefer the specific file mentioned by the user if it exists, otherwise look for others
    specific_report = os.path.join(DATA_PATH, "evaluation_report_20260221_163732.json")

    report_file = None
    if os.path.exists(specific_report):
        report_file = specific_report
    else:
        # Fallback to finding the latest report
        report_files = glob.glob(os.path.join(DATA_PATH, "evaluation_report_*.json"))
        if report_files:
            report_file = max(report_files, key=os.path.getctime)

    if not report_file:
        print("No evaluation report found.")
        return metrics_data

    try:
        with open(report_file, "r") as f:
            report_data = json.load(f)

        if (
            "classification" in report_data
            and "results" in report_data["classification"]
        ):
            metrics_data["classification"] = report_data["classification"]["results"]

        if "regression" in report_data and "results" in report_data["regression"]:
            metrics_data["regression"] = report_data["regression"]["results"]

    except Exception as e:
        print(f"Error loading evaluation report: {e}")

    return metrics_data
