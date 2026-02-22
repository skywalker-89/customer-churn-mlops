#!/usr/bin/env python3
"""
Standalone Model Visualization Script

This script can be run independently to create visualizations from existing
model evaluation results. It doesn't require running the full DAG.

Usage:
    python standalone_visualization.py
    
Or with specific options:
    python standalone_visualization.py --clf-only    # Classification only
    python standalone_visualization.py --reg-only    # Regression only
    python standalone_visualization.py --output-dir /custom/path
"""

import argparse
import os
import sys
from pathlib import Path

# Add the dags directory to Python path
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dags"))

def main():
    parser = argparse.ArgumentParser(description='Generate model performance visualizations')
    parser.add_argument('--clf-only', action='store_true', 
                       help='Generate only classification visualizations')
    parser.add_argument('--reg-only', action='store_true',
                       help='Generate only regression visualizations')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory for visualizations')
    parser.add_argument('--list-reports', action='store_true',
                       help='List available evaluation reports')
    
    args = parser.parse_args()
    
    # Import visualization functions
    try:
        from model_visualization import (
            create_classification_visualizations,
            create_regression_visualizations
        )
    except ImportError as e:
        print(f"❌ Error importing visualization module: {e}")
        print("Make sure you're running this from the correct directory")
        sys.exit(1)
    
    # Set custom output directory if specified
    if args.output_dir:
        os.environ['CUSTOM_VIZ_OUTPUT_DIR'] = args.output_dir
    
    print("🚀 Model Performance Visualization Tool")
    print("=" * 50)
    
    if args.list_reports:
        reports_dir = Path("/opt/airflow/reports/evaluation")
        if reports_dir.exists():
            print("📊 Available evaluation reports:")
            for report in sorted(reports_dir.glob("evaluation_report_*.json")):
                print(f"  - {report.name}")
        else:
            print("❌ No reports directory found")
        return
    
    # Determine which visualizations to create
    if args.clf_only:
        print("📊 Generating classification visualizations only...")
        viz_dir = create_classification_visualizations()
        print(f"✅ Classification visualizations saved to: {viz_dir}")
    elif args.reg_only:
        print("📈 Generating regression visualizations only...")
        viz_dir = create_regression_visualizations()
        print(f"✅ Regression visualizations saved to: {viz_dir}")
    else:
        print("🎨 Generating comprehensive visualizations...")
        clf_viz_dir = create_classification_visualizations()
        reg_viz_dir = create_regression_visualizations()
        print(f"✅ Classification visualizations: {clf_viz_dir}")
        print(f"✅ Regression visualizations: {reg_viz_dir}")
    
    print("\n🎉 Visualization complete!")
    print("\nTo view the visualizations:")
    print("1. Check the output directories listed above")
    print("2. Look for .png files containing the charts")
    print("3. The files are organized by model type and visualization type")

if __name__ == "__main__":
    main()