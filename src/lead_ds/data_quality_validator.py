"""
Data Quality Validation Utilities

This module provides comprehensive data quality checks for the Maven Fuzzy Factory dataset.

Checks include:
1. Null value detection
2. Schema validation
3. Data drift monitoring
4. Data type validation
5. Value range checks
"""

import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from minio import Minio
import json
from pathlib import Path

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"


class DataQualityValidator:
    """Comprehensive data quality validation"""
    
    def __init__(self, bucket_name, file_name):
        self.bucket_name = bucket_name
        self.file_name = file_name
        self.minio_client = Minio(
            MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
        )
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'file': f"{bucket_name}/{file_name}",
            'checks': {},
            'passed': True,
            'errors': [],
            'warnings': []
        }
    
    def load_data(self):
        """Load data from MinIO"""
        print(f"📥 Loading {self.file_name} from {self.bucket_name}...")
        try:
            response = self.minio_client.get_object(self.bucket_name, self.file_name)
            df = pd.read_parquet(BytesIO(response.read()))
            response.close()
            response.release_conn()
            print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            self.validation_results['passed'] = False
            self.validation_results['errors'].append(f"Failed to load data: {str(e)}")
            raise
    
    def check_null_values(self, df):
        """Check for null values in critical columns"""
        print("\n🔍 Checking for null values...")
        
        null_counts = df.isnull().sum()
        null_percentage = (null_counts / len(df) * 100).round(2)
        
        # Critical columns that should not have nulls
        critical_columns = []
        if 'website_session_id' in df.columns:
            critical_columns.append('website_session_id')
        if 'created_at' in df.columns:
            critical_columns.append('created_at')
        if 'user_id' in df.columns:
            critical_columns.append('user_id')
        
        # Check critical columns
        critical_nulls = {}
        for col in critical_columns:
            if null_counts[col] > 0:
                critical_nulls[col] = int(null_counts[col])
                self.validation_results['errors'].append(
                    f"CRITICAL: {col} has {null_counts[col]} null values ({null_percentage[col]}%)"
                )
                self.validation_results['passed'] = False
        
        # Check all columns for warnings
        warning_threshold = 5.0  # Warn if > 5% nulls
        for col in df.columns:
            if null_percentage[col] > warning_threshold and col not in critical_columns:
                self.validation_results['warnings'].append(
                    f"Column '{col}' has {null_percentage[col]}% null values"
                )
        
        self.validation_results['checks']['null_values'] = {
            'total_nulls': int(null_counts.sum()),
            'columns_with_nulls': [col for col in df.columns if null_counts[col] > 0],
            'critical_nulls': critical_nulls,
            'null_summary': {col: {'count': int(null_counts[col]), 'percentage': float(null_percentage[col])} 
                           for col in df.columns if null_counts[col] > 0}
        }
        
        if null_counts.sum() == 0:
            print("   ✅ No null values found")
        else:
            print(f"   ⚠️  Found {null_counts.sum():,} null values across {(null_counts > 0).sum()} columns")
        
        return null_counts
    
    def check_schema(self, df, expected_schema=None):
        """Validate schema matches expectations"""
        print("\n🔍 Validating schema...")
        
        actual_columns = set(df.columns)
        actual_dtypes = df.dtypes.to_dict()
        
        # Define expected schemas for different files
        schemas = {
            'website_sessions.parquet': {
                'website_session_id': 'int64',
                'created_at': 'datetime64[ns]',
                'user_id': 'int64',
                'is_repeat_session': 'int64',
                'utm_source': 'object',
                'utm_campaign': 'object',
                'utm_content': 'object',
                'device_type': 'object',
                'http_referer': 'object'
            },
            'orders.parquet': {
                'order_id': 'int64',
                'created_at': 'datetime64[ns]',
                'website_session_id': 'int64',
                'user_id': 'int64',
                'price_usd': 'float64'
            },
            'website_pageviews.parquet': {
                'website_pageview_id': 'int64',
                'created_at': 'datetime64[ns]',
                'website_session_id': 'int64',
                'pageview_url': 'object'
            },
            'training_data.parquet': {
                'is_repeat_session': ['int64', 'bool'],
                'hour_of_day': 'int64',
                'is_weekend': ['int64', 'bool'],
                'engagement_depth': 'int64',
                'is_ordered': 'int64',
                'revenue': 'float64'
            }
        }
        
        expected_schema = schemas.get(self.file_name, expected_schema)
        
        if expected_schema:
            expected_columns = set(expected_schema.keys())
            
            # Check for missing columns
            missing = expected_columns - actual_columns
            if missing:
                for col in missing:
                    self.validation_results['errors'].append(f"Missing required column: {col}")
                    self.validation_results['passed'] = False
            
            # Check for extra columns (warning only)
            extra = actual_columns - expected_columns
            if extra:
                self.validation_results['warnings'].append(f"Unexpected columns found: {list(extra)}")
            
            # Check data types
            dtype_mismatches = []
            for col, expected_dtype in expected_schema.items():
                if col in actual_columns:
                    actual_dtype = str(actual_dtypes[col])
                    # Allow for multiple valid types
                    if isinstance(expected_dtype, list):
                        if not any(actual_dtype.startswith(str(dt)) for dt in expected_dtype):
                            dtype_mismatches.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
                    else:
                        if not actual_dtype.startswith(str(expected_dtype)):
                            dtype_mismatches.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
            
            if dtype_mismatches:
                for mismatch in dtype_mismatches:
                    self.validation_results['warnings'].append(f"Data type mismatch: {mismatch}")
            
            self.validation_results['checks']['schema'] = {
                'expected_columns': len(expected_columns),
                'actual_columns': len(actual_columns),
                'missing_columns': list(missing),
                'extra_columns': list(extra),
                'dtype_mismatches': dtype_mismatches
            }
            
            if not missing and not dtype_mismatches:
                print("   ✅ Schema validation passed")
            else:
                print(f"   ⚠️  Schema issues found")
        else:
            print("   ℹ️  No expected schema defined, recording actual schema")
            self.validation_results['checks']['schema'] = {
                'columns': list(actual_columns),
                'dtypes': {k: str(v) for k, v in actual_dtypes.items()}
            }
        
        return actual_columns
    
    def check_data_ranges(self, df):
        """Check if numerical columns are within expected ranges"""
        print("\n🔍 Checking data ranges...")
        
        range_issues = []
        
        # Check specific columns based on file type
        if 'price_usd' in df.columns:
            # Revenue should be positive
            negative_revenue = (df['price_usd'] < 0).sum()
            if negative_revenue > 0:
                range_issues.append(f"Found {negative_revenue} negative price values")
                self.validation_results['errors'].append(f"Invalid data: {negative_revenue} negative prices")
                self.validation_results['passed'] = False
            
            # Check for unrealistic values (e.g., > $10,000 for a toy store)
            max_price = df['price_usd'].max()
            if max_price > 10000:
                self.validation_results['warnings'].append(f"Unusually high price detected: ${max_price:.2f}")
        
        if 'hour_of_day' in df.columns:
            invalid_hours = ((df['hour_of_day'] < 0) | (df['hour_of_day'] > 23)).sum()
            if invalid_hours > 0:
                range_issues.append(f"Found {invalid_hours} invalid hour values")
                self.validation_results['errors'].append(f"Invalid data: {invalid_hours} invalid hours (must be 0-23)")
                self.validation_results['passed'] = False
        
        if 'engagement_depth' in df.columns:
            negative_engagement = (df['engagement_depth'] < 0).sum()
            if negative_engagement > 0:
                range_issues.append(f"Found {negative_engagement} negative engagement values")
                self.validation_results['errors'].append(f"Invalid data: {negative_engagement} negative engagement depths")
                self.validation_results['passed'] = False
        
        self.validation_results['checks']['data_ranges'] = {
            'issues_found': len(range_issues),
            'issues': range_issues
        }
        
        if not range_issues:
            print("   ✅ All values within expected ranges")
        else:
            print(f"   ❌ Found {len(range_issues)} range issues")
        
        return range_issues
    
    def check_data_drift(self, df, baseline_stats_path='data_quality/baseline_stats.json'):
        """Monitor data drift by comparing against baseline statistics"""
        print("\n🔍 Checking for data drift...")
        
        # Calculate current statistics
        current_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            current_stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        # Try to load baseline statistics
        baseline_exists = False
        try:
            if Path(baseline_stats_path).exists():
                with open(baseline_stats_path, 'r') as f:
                    baseline_stats = json.load(f)
                baseline_exists = True
        except Exception as e:
            print(f"   ℹ️  No baseline found, creating new baseline")
            baseline_exists = False
        
        drift_detected = []
        
        if baseline_exists:
            # Compare current stats to baseline
            drift_threshold = 0.2  # 20% change triggers warning
            
            for col in current_stats.keys():
                if col in baseline_stats:
                    for metric in ['mean', 'std']:
                        baseline_val = baseline_stats[col][metric]
                        current_val = current_stats[col][metric]
                        
                        if baseline_val != 0:
                            pct_change = abs((current_val - baseline_val) / baseline_val)
                            
                            if pct_change > drift_threshold:
                                drift_detected.append({
                                    'column': col,
                                    'metric': metric,
                                    'baseline': baseline_val,
                                    'current': current_val,
                                    'change_pct': f"{pct_change * 100:.2f}%"
                                })
                                self.validation_results['warnings'].append(
                                    f"Drift detected in {col}.{metric}: {pct_change*100:.1f}% change"
                                )
            
            if drift_detected:
                print(f"   ⚠️  Detected drift in {len(drift_detected)} metrics")
            else:
                print("   ✅ No significant drift detected")
        else:
            # Save current stats as baseline
            Path(baseline_stats_path).parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_stats_path, 'w') as f:
                json.dump(current_stats, f, indent=2)
            print(f"   ✅ Baseline statistics saved to {baseline_stats_path}")
        
        self.validation_results['checks']['data_drift'] = {
            'baseline_exists': baseline_exists,
            'drift_detected': len(drift_detected),
            'drifts': drift_detected,
            'current_stats': current_stats
        }
        
        return drift_detected
    
    def run_all_checks(self):
        """Run all validation checks"""
        print("=" * 60)
        print("  DATA QUALITY VALIDATION")
        print(f"  File: {self.bucket_name}/{self.file_name}")
        print("=" * 60)
        
        try:
            # Load data
            df = self.load_data()
            
            # Run checks
            self.check_null_values(df)
            self.check_schema(df)
            self.check_data_ranges(df)
            self.check_data_drift(df)
            
            # Summary
            print("\n" + "=" * 60)
            if self.validation_results['passed']:
                print("✅ DATA QUALITY VALIDATION PASSED")
            else:
                print("❌ DATA QUALITY VALIDATION FAILED")
            
            print(f"\n📊 Summary:")
            print(f"   Errors: {len(self.validation_results['errors'])}")
            print(f"   Warnings: {len(self.validation_results['warnings'])}")
            
            if self.validation_results['errors']:
                print("\n❌ Errors:")
                for error in self.validation_results['errors']:
                    print(f"   - {error}")
            
            if self.validation_results['warnings']:
                print("\n⚠️  Warnings:")
                for warning in self.validation_results['warnings']:
                    print(f"   - {warning}")
            
            print("=" * 60)
            
            return self.validation_results
            
        except Exception as e:
            print(f"\n❌ Validation failed with exception: {str(e)}")
            self.validation_results['passed'] = False
            self.validation_results['errors'].append(f"Exception: {str(e)}")
            return self.validation_results
    
    def save_report(self, output_dir='reports/data_quality'):
        """Save validation report to file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{output_dir}/validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        return report_file


if __name__ == "__main__":
    # Example usage
    validator = DataQualityValidator("processed-data", "training_data.parquet")
    results = validator.run_all_checks()
    validator.save_report()
