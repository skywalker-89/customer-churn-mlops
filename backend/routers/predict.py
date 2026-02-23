from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from core.minio_client import minio_client
import sys
import os

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

router = APIRouter()


# --- Schemas ---
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_name: str
    model_type: str  # "classification" or "regression"


class CustomerFilter(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    income_level: Optional[str] = None
    min_purchase_value: Optional[float] = None


# --- Helpers ---
def load_and_prep_data():
    """
    Load training data. Tries MinIO first, then local file, then mock data.
    """
    # 1. Try MinIO
    df = minio_client.get_data("training_data.parquet")

    # 2. Try local file
    if df is None:
        try:
            local_path = os.path.join(
                os.path.dirname(__file__), "../../data/processed/training_data.parquet"
            )
            if os.path.exists(local_path):
                df = pd.read_parquet(local_path)
        except Exception as e:
            print(f"Error loading local parquet: {e}")

    # 3. Create mock data if nothing found (for development/demo)
    if df is None:
        print("Warning: Could not load data. Generating mock data for demonstration.")
        # Create a mock dataframe with all expected columns
        data = {
            "age": np.random.randint(18, 80, 100),
            "income": np.random.randint(20000, 150000, 100),
            "total_spent": np.random.uniform(100, 5000, 100),
            "num_purchases": np.random.randint(1, 50, 100),
            "avg_purchase": np.random.uniform(20, 500, 100),
            "days_since_last_purchase": np.random.randint(1, 365, 100),
            "website_visits": np.random.randint(1, 100, 100),
            "has_complained": np.random.randint(0, 2, 100),
            "is_active_member": np.random.randint(0, 2, 100),
            "total_sales": np.random.uniform(100, 5000, 100),
            "churned": np.random.randint(0, 2, 100),
            "clv_per_year": np.random.uniform(50, 1000, 100),
            # New features for engineering
            "quantity": np.random.randint(1, 20, 100),
            "unit_price": np.random.uniform(10, 500, 100),
            "discount_applied": np.random.uniform(0, 0.3, 100),
            "online_purchases": np.random.randint(0, 30, 100),
            "in_store_purchases": np.random.randint(0, 30, 100),
            "email_subscriptions": np.random.randint(0, 2, 100),
            "app_usage_frequency": np.random.choice(["Low", "Medium", "High"], 100),
            "social_media_engagement": np.random.choice(["Low", "Medium", "High"], 100),
            "product_return_rate": np.random.uniform(0, 0.5, 100),
            "customer_support_calls": np.random.randint(0, 10, 100),
            # Geo data
            "customer_city": np.random.choice(
                ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 100
            ),
            "customer_state": np.random.choice(["NY", "CA", "IL", "TX", "AZ"], 100),
            "store_location": np.random.choice(
                ["Downtown", "Mall", "Airport", "Suburb"], 100
            ),
            "gender": np.random.choice(["Male", "Female", "Other"], 100),
            "education_level": np.random.choice(
                ["High School", "Bachelor", "Master", "PhD"], 100
            ),
            "marital_status": np.random.choice(["Married", "Single", "Divorced"], 100),
            "product_category": np.random.choice(
                ["Electronics", "Clothing", "Home", "Books"], 100
            ),
            "payment_method": np.random.choice(
                ["Credit Card", "PayPal", "Debit Card"], 100
            ),
        }
        df = pd.DataFrame(data)

    return df


def _prepare_features_for_inference(model, df):
    """
    Helper to prepare features for inference from a dataframe.
    Handles different model expectations and data types.
    Applies feature engineering consistent with training pipeline.
    """
    try:
        # Create a copy to avoid modifying original
        X = df.copy()

        # --- Feature Engineering ---
        # 1. Quantity * Price (Most critical feature)
        if "quantity" in X.columns and "unit_price" in X.columns:
            X["quantity_times_price"] = X["quantity"] * X["unit_price"]
        elif "total_sales" in X.columns:
            # Fallback
            X["quantity_times_price"] = X["total_sales"]
        elif "total_spent" in X.columns:
            # Fallback
            X["quantity_times_price"] = X["total_spent"]
        else:
            X["quantity_times_price"] = 0.0

        # 2. Engagement Score
        # Formula: (app_usage_Medium + app_usage_High + social_media_engagement_High + email_subscriptions) / 4
        # Note: We need to do one-hot encoding first or handle it manually here
        # Let's handle it manually to be safe before one-hot encoding
        app_usage_score = (
            X["app_usage_frequency"].apply(
                lambda x: 1.0 if x in ["Medium", "High"] else 0.0
            )
            if "app_usage_frequency" in X.columns
            else 0.0
        )

        social_score = (
            X["social_media_engagement"].apply(lambda x: 1.0 if x == "High" else 0.0)
            if "social_media_engagement" in X.columns
            else 0.0
        )

        email_score = (
            X["email_subscriptions"] if "email_subscriptions" in X.columns else 0.0
        )

        # We need another component to match 4 parts, likely app_usage_High separately?
        # Re-reading docs:
        # df.get('app_usage_Medium', 0) + df.get('app_usage_High', 0) + df.get('social_media_engagement_High', 0) + df.get('email_subscriptions', 0)
        # So app_usage counts TWICE if High? No, if High, app_usage_Medium is 0.
        # If Medium, app_usage_Medium is 1, High is 0.
        # So "Medium or High" logic is correct for app usage contribution.

        X["engagement_score"] = (app_usage_score + social_score + email_score) / 4.0

        # 3. Recency Ratio
        if "days_since_last_purchase" in X.columns:
            X["recency_ratio"] = X["days_since_last_purchase"] / 365.0
        else:
            X["recency_ratio"] = 0.0

        # 4. Online Preference
        if "online_purchases" in X.columns and "in_store_purchases" in X.columns:
            total_purchases = X["online_purchases"] + X["in_store_purchases"] + 1
            X["online_preference"] = X["online_purchases"] / total_purchases
        elif "online_purchases" in X.columns and "num_purchases" in X.columns:
            X["online_preference"] = X["online_purchases"] / (X["num_purchases"] + 1)
        else:
            X["online_preference"] = 0.5  # Default

        # 5. Value per transaction
        if "total_spent" in X.columns and "num_purchases" in X.columns:
            X["value_per_transaction"] = X["total_spent"] / (
                X["num_purchases"].replace(0, 1)
            )
        else:
            X["value_per_transaction"] = 0.0

        # --- One-Hot Encoding ---
        # Identify categorical columns
        categorical_cols = [
            "gender",
            "store_location",
            "customer_city",
            "store_city",
            "app_usage_frequency",
            "social_media_engagement",
            "education_level",
            "marital_status",
            "product_category",
            "payment_method",
            "income_level",
        ]

        # Get dummies
        existing_cats = [col for col in categorical_cols if col in X.columns]
        if existing_cats:
            X = pd.get_dummies(X, columns=existing_cats, drop_first=True)

        # Drop target columns if present
        X = X.drop(columns=["total_sales", "churned", "clv_per_year"], errors="ignore")

        # --- Align with Model Features ---
        if hasattr(model, "feature_names"):
            expected_features = model.feature_names
        elif hasattr(model, "n_features_in_"):
            # If we can't get names, we can't align by name.
            # We have to hope column order is somewhat preserved or use what we have.
            # But usually for scratch models we added feature_names attribute.
            # For sklearn models, they have feature_names_in_
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
            else:
                # Fallback: return X as is, hoping for the best
                return X.fillna(0)
        else:
            return X.fillna(0)

        # Create final dataframe with expected features
        final_X = pd.DataFrame(index=X.index)

        for feature in expected_features:
            if feature in X.columns:
                final_X[feature] = X[feature]
            else:
                # Handle missing features
                final_X[feature] = 0.0

        return final_X.fillna(0)

    except Exception as e:
        print(f"Error preparing features: {e}")
        # Final fallback: return whatever numeric data we have
        return df.select_dtypes(include=[np.number]).fillna(0)


def convert_categorical_to_numeric(input_features, expected_features):
    """
    Convert categorical input features to match the model's expected format.
    This handles the conversion from frontend form data to model-ready features.
    """
    # Create a dictionary to hold all features, initialized to 0
    processed_features = {feature: 0 for feature in expected_features}

    # Direct numeric mappings
    numeric_mappings = {
        "age": "age",
        "income": "income",
        "total_spent": "total_spent",
        "num_purchases": "num_purchases",
        "avg_purchase": "avg_purchase",
        "days_since_last_purchase": "days_since_last_purchase",
        "website_visits": "website_visits",
        "has_complained": "has_complained",
        "is_active_member": "is_active_member",
        "online_purchases": "online_purchases",
        "in_store_purchases": "in_store_purchases",
        "email_subscriptions": "email_subscriptions",
        "distance_to_store": "distance_to_store",
        "product_return_rate": "product_return_rate",
        "quantity": "quantity",
        "unit_price": "unit_price",
        "discount_applied": "discount_applied",
    }

    # Map direct numeric features
    for frontend_key, model_key in numeric_mappings.items():
        if frontend_key in input_features:
            processed_features[model_key] = float(input_features[frontend_key])

    # Extended mappings for 43-feature models
    if "total_transactions" in expected_features and "num_purchases" in input_features:
        processed_features["total_transactions"] = float(
            input_features["num_purchases"]
        )

    if (
        "avg_transaction_value" in expected_features
        and "avg_purchase" in input_features
    ):
        processed_features["avg_transaction_value"] = float(
            input_features["avg_purchase"]
        )

    if "avg_purchase_value" in expected_features and "avg_purchase" in input_features:
        # Map avg_purchase to avg_purchase_value if present
        processed_features["avg_purchase_value"] = float(input_features["avg_purchase"])

    if "loyalty_program" in expected_features and "is_active_member" in input_features:
        processed_features["loyalty_program"] = float(
            input_features["is_active_member"]
        )

    if (
        "customer_support_calls" in expected_features
        and "has_complained" in input_features
    ):
        # If has complained, assume 1 call (or keep default if 0)
        # Note: input_features["has_complained"] is 0 or 1
        processed_features["customer_support_calls"] = float(
            input_features["has_complained"]
        )

    if (
        "recency_ratio" in expected_features
        and "days_since_last_purchase" in input_features
    ):
        # Heuristic: ratio of days since last purchase to 1 year
        processed_features["recency_ratio"] = (
            float(input_features["days_since_last_purchase"]) / 365.0
        )

    # Engineered feature: recency_score (1 / (days + 1))
    if (
        "recency_score" in expected_features
        and "days_since_last_purchase" in input_features
    ):
        processed_features["recency_score"] = 1.0 / (
            float(input_features["days_since_last_purchase"]) + 1.0
        )

    # Engineered feature: quantity_times_price
    # This is the most critical feature for regression (R2=0.9988)
    if "quantity_times_price" in expected_features:
        if "quantity" in input_features and "unit_price" in input_features:
            quantity = float(input_features["quantity"])
            unit_price = float(input_features["unit_price"])
            discount = 0.0
            if "discount_applied" in input_features:
                discount = float(input_features["discount_applied"])

            processed_features["quantity_times_price"] = (
                quantity * unit_price * (1 - discount)
            )
        elif "total_spent" in input_features:
            # Fallback: Use total_spent as proxy if raw quantity/price not available
            processed_features["quantity_times_price"] = float(
                input_features["total_spent"]
            )

    # Engineered feature: value_per_transaction
    if "value_per_transaction" in expected_features:
        total_sales = 0.0
        total_transactions = 1.0

        if "total_spent" in input_features:
            total_sales = float(input_features["total_spent"])
        elif "quantity_times_price" in processed_features:
            total_sales = processed_features["quantity_times_price"]

        if "num_purchases" in input_features:
            total_transactions = max(1.0, float(input_features["num_purchases"]))

        processed_features["value_per_transaction"] = total_sales / total_transactions

    # Engineered feature: engagement_score
    if "engagement_score" in expected_features:
        # Formula: (app_usage_Medium + app_usage_High + social_media_engagement_High + email_subscriptions) / 4

        # Component 1 & 2: App usage (Medium or High)
        # In the original one-hot encoding, app_usage_Medium=1 if Medium, app_usage_High=1 if High.
        # Since they are mutually exclusive, (app_usage_Medium + app_usage_High) is 1 if usage is either Medium or High.
        app_usage_component = 0.0
        if "app_usage_frequency" in input_features:
            usage = input_features["app_usage_frequency"]
            if usage in ["Medium", "High"]:
                app_usage_component = 1.0
        elif "website_visits" in input_features:
            # Fallback
            if float(input_features["website_visits"]) > 5:
                app_usage_component = 1.0

        # Component 3: Social media High
        social_component = 0.0
        if "social_media_engagement" in input_features:
            if input_features["social_media_engagement"] == "High":
                social_component = 1.0

        # Component 4: Email subscriptions
        email_component = 0.0
        if "email_subscriptions" in input_features:
            email_component = float(input_features["email_subscriptions"])
        elif (
            "is_active_member" in input_features and input_features["is_active_member"]
        ):
            email_component = 1.0

        processed_features["engagement_score"] = (
            app_usage_component + social_component + email_component
        ) / 4.0

    # Engineered feature: online_preference and engagement_ratio
    if (
        "online_preference" in expected_features
        or "engagement_ratio" in expected_features
    ):
        online = 0.0
        in_store = 0.0

        if "online_purchases" in input_features:
            online = float(input_features["online_purchases"])
        elif "num_purchases" in input_features:
            # Split num_purchases based on store_location presence?
            # If store_location is selected, assume more in-store?
            # For now, split 50/50 as fallback
            online = float(input_features["num_purchases"]) * 0.5

        if "in_store_purchases" in input_features:
            in_store = float(input_features["in_store_purchases"])
        elif "num_purchases" in input_features:
            in_store = float(input_features["num_purchases"]) * 0.5

        total = online + in_store
        if total == 0:
            total = 1.0  # Avoid division by zero

        preference = online / total

        if "online_preference" in expected_features:
            processed_features["online_preference"] = preference

        if "engagement_ratio" in expected_features:
            processed_features["engagement_ratio"] = preference

    # Handle categorical features with one-hot encoding
    # Gender mapping
    if "gender" in input_features:
        gender = input_features["gender"]
        if gender == "Male":
            processed_features["gender_Male"] = 1
        elif gender == "Other":
            processed_features["gender_Other"] = 1
        # Female is the baseline (all gender_* features = 0)

    # Location mapping
    if "store_location" in input_features:
        store_location = input_features["store_location"]
        if f"store_location_{store_location}" in expected_features:
            processed_features[f"store_location_{store_location}"] = 1

    if "customer_city" in input_features:
        customer_city = input_features["customer_city"]
        if f"customer_city_{customer_city}" in expected_features:
            processed_features[f"customer_city_{customer_city}"] = 1

    if "store_city" in input_features:
        store_city = input_features["store_city"]
        if f"store_city_{store_city}" in expected_features:
            processed_features[f"store_city_{store_city}"] = 1

    # Income bracket mapping
    if "income" in input_features:
        income = float(input_features["income"])
        if income < 30000:
            processed_features["income_bracket_Low"] = 1
        elif income < 70000:
            processed_features["income_bracket_Medium"] = 1
        # High income is the baseline

    # Purchase frequency mapping
    if "num_purchases" in input_features:
        num_purchases = float(input_features["num_purchases"])
        if num_purchases >= 50:
            processed_features["purchase_frequency_Weekly"] = 1
        elif num_purchases >= 20:
            processed_features["purchase_frequency_Monthly"] = 1
        else:
            processed_features["purchase_frequency_Yearly"] = 1

    # App usage mapping
    if "app_usage_frequency" in input_features:
        usage = input_features["app_usage_frequency"]
        if usage == "Low":
            if "app_usage_Low" in expected_features:
                processed_features["app_usage_Low"] = 1
        elif usage == "Medium":
            if "app_usage_Medium" in expected_features:
                processed_features["app_usage_Medium"] = 1
        # High is baseline (0, 0)

    # Social media mapping
    if "social_media_engagement" in input_features:
        engagement = input_features["social_media_engagement"]
        if engagement == "Low":
            if "social_media_engagement_Low" in expected_features:
                processed_features["social_media_engagement_Low"] = 1
        elif engagement == "Medium":
            if "social_media_engagement_Medium" in expected_features:
                processed_features["social_media_engagement_Medium"] = 1
        # High is baseline (0, 0)

    # Education Level mapping
    if "education_level" in input_features:
        edu = input_features["education_level"]
        if edu == "High School":
            if "education_level_High School" in expected_features:
                processed_features["education_level_High School"] = 1
        elif edu == "Bachelor":
            if "education_level_Bachelor" in expected_features:
                processed_features["education_level_Bachelor"] = 1
        elif edu == "Master":
            if "education_level_Master" in expected_features:
                processed_features["education_level_Master"] = 1
        elif edu == "PhD":
            if "education_level_PhD" in expected_features:
                processed_features["education_level_PhD"] = 1

    # Marital Status mapping
    if "marital_status" in input_features:
        status = input_features["marital_status"]
        if status == "Married":
            if "marital_status_Married" in expected_features:
                processed_features["marital_status_Married"] = 1
        elif status == "Single":
            if "marital_status_Single" in expected_features:
                processed_features["marital_status_Single"] = 1
        elif status == "Divorced":
            if "marital_status_Divorced" in expected_features:
                processed_features["marital_status_Divorced"] = 1

    # Product Category mapping
    if "product_category" in input_features:
        cat = input_features["product_category"]
        if f"product_category_{cat}" in expected_features:
            processed_features[f"product_category_{cat}"] = 1

    # Payment Method mapping
    if "payment_method" in input_features:
        method = input_features["payment_method"]
        if f"payment_method_{method}" in expected_features:
            processed_features[f"payment_method_{method}"] = 1

    # Default some common features that might be missing
    # Set reasonable defaults for missing features
    defaults = {
        "loyalty_program": 0,  # Default to No
        "membership_years": 1,
        "number_of_children": 0,
        "transaction_date": 2021.0,  # Use year as proxy
        "quantity": 5.0,
        "unit_price": 100.0,
        "discount_applied": 0.1,
        "transaction_hour": 12.0,
        "week_of_year": 26.0,
        "month_of_year": 6.0,
        "avg_purchase_value": 200.0,
        "last_purchase_date": 2021.0,
        "avg_discount_used": 0.15,
        "online_purchases": 10.0,
        "in_store_purchases": 15.0,
        "avg_items_per_transaction": 3.0,
        "avg_transaction_value": 150.0,
        "total_returned_items": 1.0,
        "total_returned_value": 50.0,
        "total_transactions": 20.0,
        "total_items_purchased": 60.0,
        "total_discounts_received": 100.0,
        "avg_spent_per_category": 200.0,
        "max_single_purchase_value": 500.0,
        "min_single_purchase_value": 10.0,
        "product_rating": 3.5,
        "product_review_count": 100.0,
        "product_stock": 50.0,
        "product_return_rate": 0.1,
        "product_weight": 1.0,
        "product_manufacture_date": 2020.0,
        "product_expiry_date": 2023.0,
        "product_shelf_life": 365.0,
        "promotion_start_date": 2021.0,
        "promotion_end_date": 2021.0,
        "customer_zip_code": 10001.0,
        "store_zip_code": 10001.0,
        "distance_to_store": 10.0,
        "holiday_season": 0.0,
        "weekend": 0.0,
        "customer_support_calls": 2.0,
        "email_subscriptions": 1.0,
    }

    # Apply defaults for missing features
    for feature, default_value in defaults.items():
        if feature in processed_features and processed_features[feature] == 0:
            processed_features[feature] = default_value

    return processed_features


# --- Endpoints ---


@router.get("/models")
async def get_available_models():
    """List available models for classification and regression"""
    try:
        # Get list of models from MinIO
        # This assumes models are saved as "{model_name}_latest.pkl"
        # We need to access the client directly as list_models returns full object names
        # But wait, minio_client.list_models() returns object names
        model_files = minio_client.list_models()

        # Extract model names
        models = []
        for m in model_files:
            if m.endswith("_latest.pkl"):
                models.append(m.replace("_latest.pkl", ""))
            elif m.endswith(".pkl"):
                models.append(m.replace(".pkl", ""))
            else:
                models.append(m)

        # Categorize
        classification_models = []
        regression_models = []

        for m in models:
            m_lower = m.lower()
            if "classification" in m_lower:
                classification_models.append(m)
            elif "regression" in m_lower:
                regression_models.append(m)
            elif "sklearn" in m_lower:
                # Handle sklearn models based on known types
                if "random_forest" in m_lower:
                    classification_models.append(m)
                elif "xgboost" in m_lower:
                    regression_models.append(m)
                else:
                    # Default fallback for unknown sklearn models
                    classification_models.append(m)
                    regression_models.append(m)
            else:
                # Uncategorized - add to both to be safe
                classification_models.append(m)
                regression_models.append(m)

        # Sort for better UX
        classification_models.sort()
        regression_models.sort()

        return {
            "classification": classification_models,
            "regression": regression_models,
        }
    except Exception as e:
        print(f"Error listing models: {e}")
        # Fallback if MinIO fails or is empty
        return {
            "classification": [
                "logistic_regression_classification",
                "random_forest_classification",
                "decision_tree_classification",
                "svm_classification",
                "xgboost_classification",
            ],
            "regression": [
                "linear_regression_regression",
                "random_forest_regression",
                "decision_tree_regression",
            ],
        }


@router.get("/features")
async def get_feature_info():
    """Get feature names and types for the frontend input form"""
    df = load_and_prep_data()
    if df is None:
        return {"error": "Could not load data"}

    # Exclude targets
    drop_cols = ["total_sales", "churned", "clv_per_year"]
    features = [c for c in df.columns if c not in drop_cols]

    # Get simple stats for numerical columns to set ranges
    stats = {}
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "type": "numeric",
            }
        else:
            stats[col] = {
                "unique": df[col].unique().tolist()[:50],  # Limit to 50 options
                "type": "categorical",
            }

    return {"features": features, "stats": stats}


@router.post("/predict")
async def predict(request: PredictionRequest):
    """Make a prediction using a specified model"""

    # 1. Load Model
    # Map frontend model name to MinIO key
    # E.g. "Random Forest" -> "randomforest_classification"
    # This mapping might need adjustment based on exact filenames in MinIO

    model = minio_client.get_model(request.model_name)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"Model {request.model_name} not found"
        )

    # 2. Prepare Features
    # Get expected features from the model's training data
    data_df = load_and_prep_data()
    if data_df is None:
        raise HTTPException(
            status_code=500, detail="Could not load training data for feature schema"
        )

    # Exclude targets to get expected feature names
    drop_cols = ["total_sales", "churned", "clv_per_year"]
    all_features = [c for c in data_df.columns if c not in drop_cols]

    # Try to get the actual expected features from the model
    try:
        # Known feature sets for fallback
        # Features from logistic regression model (43 features)
        FALLBACK_FEATURES_43 = [
            "age",
            "loyalty_program",
            "membership_years",
            "number_of_children",
            "quantity",
            "unit_price",
            "discount_applied",
            "transaction_hour",
            "week_of_year",
            "month_of_year",
            "avg_purchase_value",
            "avg_discount_used",
            "online_purchases",
            "in_store_purchases",
            "avg_items_per_transaction",
            "avg_transaction_value",
            "total_returned_items",
            "total_returned_value",
            "total_transactions",
            "total_items_purchased",
            "total_discounts_received",
            "avg_spent_per_category",
            "max_single_purchase_value",
            "min_single_purchase_value",
            "product_rating",
            "product_review_count",
            "product_stock",
            "product_return_rate",
            "product_weight",
            "product_shelf_life",
            "customer_zip_code",
            "store_zip_code",
            "distance_to_store",
            "holiday_season",
            "weekend",
            "customer_support_calls",
            "email_subscriptions",
            "website_visits",
            "days_since_last_purchase",
            "quantity_times_price",
            "engagement_score",
            "recency_ratio",
            "online_preference",
        ]

        # Check explicit feature names first (common in our scratch models)
        if hasattr(model, "feature_names") and model.feature_names:
            expected_features = list(model.feature_names)
            print(f"Using {len(expected_features)} features from model.feature_names")
        # Check sklearn standard attribute
        elif hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            print(
                f"Using {len(expected_features)} features from model.feature_names_in_"
            )
        # Check n_features_in_
        elif hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
            if n_features == 43:
                expected_features = FALLBACK_FEATURES_43
                print(f"Using fallback feature list for {n_features} features")
            else:
                expected_features = all_features[:n_features]
                print(
                    f"Model expects {n_features} features from n_features_in_ (using first {n_features} from data)"
                )
        # Check specific scratch models that we know match the standard pipeline
        elif (
            "xgboost" in request.model_name.lower()
            or "regression" in request.model_name.lower()
        ):
            # Assumption: All our scratch regression models use the standard 43 features
            expected_features = FALLBACK_FEATURES_43
            print(
                f"Using fallback feature list (43 features) for known scratch model: {request.model_name}"
            )
        else:
            # Fallback: Try to predict with empty input to see what the model expects
            # Try to predict with empty input to see what the model expects
            test_input = pd.DataFrame([{col: 0 for col in all_features}])
            model.predict(test_input.values)
            # If this works, use all features
            expected_features = all_features
            print(
                f"Using all {len(all_features)} features from training data (fallback)"
            )
    except Exception as e:
        print(f"Model error with all features: {e}")
        # Try to extract expected feature count from error
        error_str = str(e)

        # Use model's feature_names if available
        if hasattr(model, "feature_names") and model.feature_names:
            expected_features = list(model.feature_names)
            print(f"Using {len(expected_features)} features from model.feature_names")
        elif "shapes (1," in error_str and "(43,)" in error_str:
            n_features = 43
            print(f"Model expects {n_features} features from error")
            # Use only the first n_features from the training data
            expected_features = all_features[:n_features]
        elif hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
            expected_features = all_features[:n_features]
            print(f"Model expects {n_features} features from n_features_in_")
        else:
            # Default to 43 as we saw in the error
            n_features = 43
            expected_features = all_features[:n_features]
            print(f"Defaulting to {n_features} features")

    # Convert categorical input to numeric format expected by model
    processed_features = convert_categorical_to_numeric(
        request.features, expected_features
    )

    # Create DataFrame with processed features, ensuring only expected features are included
    input_df = pd.DataFrame([processed_features])

    # Ensure we only have the exact features the model expects
    final_features = {
        feature: processed_features.get(feature, 0) for feature in expected_features
    }
    final_df = pd.DataFrame([final_features])

    # Ensure all columns expected by model are present (fill missing with 0 or defaults)
    X = final_df.values

    # 3. Predict
    try:
        # Debug: Check shapes
        # print(f"Input shape: {X.shape}")
        print(f"Expected features: {len(expected_features)}")
        print(f"First 5 expected features: {expected_features[:5]}")
        if hasattr(model, "n_features_in_"):
            print(f"Model n_features_in_: {model.n_features_in_}")
        if hasattr(model, "feature_names"):
            print(
                f"Model feature_names: {len(model.feature_names) if model.feature_names else 'None'}"
            )

        prediction = model.predict(X)
        result = {}

        # Handle scalar output from some models (0-d array)
        if hasattr(prediction, "ndim") and prediction.ndim == 0:
            pred_val = prediction.item()
        elif hasattr(prediction, "__len__") and len(prediction) > 0:
            # Ensure we get the scalar value from 1-element array
            if hasattr(prediction[0], "item"):
                pred_val = prediction[0].item()
            else:
                pred_val = prediction[0]
        else:
            pred_val = prediction

        if request.model_type == "classification":
            result["prediction"] = int(pred_val)
            result["label"] = "Churn" if int(pred_val) == 1 else "Retain"
            # Try to get probability if available
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                    # Handle probability shape (1D vs 2D)
                    if hasattr(proba, "ndim") and proba.ndim == 1:
                        # 1D array [prob_0, prob_1]
                        result["probability"] = (
                            float(proba[1]) if len(proba) > 1 else float(proba[0])
                        )
                    elif hasattr(proba, "ndim") and proba.ndim == 2:
                        # 2D array [[prob_0, prob_1]]
                        result["probability"] = (
                            float(proba[0][1])
                            if proba.shape[1] > 1
                            else float(proba[0][0])
                        )
                    else:
                        # List or other format
                        if len(proba) > 0:
                            if (
                                isinstance(proba[0], (list, np.ndarray))
                                and len(proba[0]) > 1
                            ):
                                result["probability"] = float(proba[0][1])
                            elif len(proba) > 1:
                                result["probability"] = float(proba[1])
                            else:
                                result["probability"] = float(proba[0])
                        else:
                            result["probability"] = 0.5
                except Exception as e:
                    print(f"Error getting probability: {e}")
                    result["probability"] = 0.5
        else:
            result["prediction"] = float(pred_val)
            result["label"] = f"${float(pred_val):,.2f}"

        # Add model information
        result["model_used"] = request.model_name

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/what-if")
async def predict_what_if(request: PredictionRequest):
    """
    What-if analysis: Compare predictions with modified features
    Returns original prediction and predictions with feature variations
    """
    # Get original prediction first
    original_result = await predict(request)

    # Define feature variations for what-if analysis
    variations = {}

    # Get feature stats for reasonable variation ranges
    data_df = load_and_prep_data()
    if data_df is not None:
        for feature, value in request.features.items():
            if feature in data_df.columns and pd.api.types.is_numeric_dtype(
                data_df[feature]
            ):
                mean_val = data_df[feature].mean()
                std_val = data_df[feature].std()

                # Create variations: -1 std, +1 std, -2 std, +2 std
                variations[feature] = [
                    ("-2σ", max(value - 2 * std_val, data_df[feature].min())),
                    ("-1σ", max(value - 1 * std_val, data_df[feature].min())),
                    ("+1σ", min(value + 1 * std_val, data_df[feature].max())),
                    ("+2σ", min(value + 2 * std_val, data_df[feature].max())),
                ]

    # Test each variation
    what_if_results = []

    for feature, feature_variations in variations.items():
        for variation_name, variation_value in feature_variations:
            # Create modified features
            modified_features = request.features.copy()
            modified_features[feature] = variation_value

            # Create modified request
            modified_request = PredictionRequest(
                model_name=request.model_name,
                model_type=request.model_type,
                features=modified_features,
            )

            try:
                # Get prediction for this variation
                variation_result = await predict(modified_request)

                what_if_results.append(
                    {
                        "feature": feature,
                        "variation": variation_name,
                        "value": (
                            float(variation_value)
                            if isinstance(variation_value, (np.integer, np.floating))
                            else variation_value
                        ),
                        "prediction": (
                            float(variation_result["prediction"])
                            if isinstance(
                                variation_result["prediction"],
                                (np.integer, np.floating),
                            )
                            else variation_result["prediction"]
                        ),
                        "label": variation_result.get("label", ""),
                        "probability": (
                            float(variation_result.get("probability", 0))
                            if variation_result.get("probability") is not None
                            else None
                        ),
                    }
                )
            except Exception as e:
                # Skip this variation if prediction fails
                continue

    # Calculate feature impact (how much each feature affects prediction)
    feature_impacts = {}
    original_pred = original_result["prediction"]

    for feature in variations.keys():
        feature_variations = [r for r in what_if_results if r["feature"] == feature]
        if feature_variations:
            predictions = [r["prediction"] for r in feature_variations]
            impacts = [abs(pred - original_pred) for pred in predictions]
            feature_impacts[feature] = {
                "max_impact": float(max(impacts)) if impacts else 0.0,
                "avg_impact": float(sum(impacts) / len(impacts)) if impacts else 0.0,
                "sensitivity": (
                    float(max(impacts) - min(impacts)) if len(impacts) > 1 else 0.0
                ),
            }

    return {
        "original": original_result,
        "variations": what_if_results,
        "feature_impacts": feature_impacts,
        "insights": {
            "most_sensitive_feature": (
                max(
                    feature_impacts.keys(),
                    key=lambda x: feature_impacts[x]["max_impact"],
                )
                if feature_impacts
                else None
            ),
            "least_sensitive_feature": (
                min(
                    feature_impacts.keys(),
                    key=lambda x: feature_impacts[x]["max_impact"],
                )
                if feature_impacts
                else None
            ),
            "total_variations_tested": len(what_if_results),
        },
    }


@router.get("/customers/geo")
async def get_geo_data():
    """Get customer location data for mapping"""
    # Note: The processed training data might not have lat/lon or city names if they were encoded.
    # We might need to look at raw data or `retail_data.parquet` if available.

    df = None
    try:
        df = minio_client.get_data(
            "retail_data.parquet", bucket_name="raw-data"
        )  # Try raw bucket first
    except:
        pass

    if df is None:
        # Try processed and see if we can use store_location or similar
        df = load_and_prep_data()

    if df is None:
        return {"error": "Data not found"}

    # Check for location columns
    # Based on features list: 'customer_city', 'customer_state'

    # If we have processed data, we might not have raw city names.
    # Check if we have the raw columns
    has_geo_cols = "customer_city" in df.columns and "customer_state" in df.columns

    if not has_geo_cols:
        # If we don't have raw geo columns, we should generate mock geo data
        # based on the existing data or just return mock data
        print("Warning: No geographic columns found in data. Generating mock geo data.")

        # Create mock geo data
        cities = [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Philadelphia",
            "San Antonio",
            "San Diego",
            "Dallas",
            "San Jose",
        ]
        states = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA"]

        mock_data = []
        for i in range(len(cities)):
            mock_data.append(
                {
                    "city": cities[i],
                    "country": states[i],
                    "customers": np.random.randint(50, 500),
                    "avg_revenue": float(np.random.uniform(2000, 10000)),
                    "churn_rate": float(np.random.uniform(0.1, 0.4)),
                }
            )
        return mock_data

    # Group by city/state to get aggregates
    try:
        # Create a copy to avoid modifying original
        df_geo = df.copy()

        # Check if churned column exists and is numeric
        if "churned" in df_geo.columns and pd.api.types.is_numeric_dtype(
            df_geo["churned"]
        ):
            agg_dict = {"churned": "mean"}
        else:
            agg_dict = {}

        # Add total_sales aggregation if column exists and is numeric
        if "total_sales" in df_geo.columns and pd.api.types.is_numeric_dtype(
            df_geo["total_sales"]
        ):
            agg_dict["total_sales"] = "sum"

        # Add customer count
        agg_dict["customer_city"] = "count"

        grouped = (
            df_geo.groupby(["customer_city", "customer_state"])
            .agg(agg_dict)
            .rename(columns={"customer_city": "customer_count"})
            .reset_index()
        )

        # Rename columns to match frontend expectations
        column_mapping = {
            "customer_city": "city",
            "customer_state": "country",  # Using state as country for frontend compatibility
            "total_sales": "avg_revenue",
            "customer_count": "customers",
        }
        grouped = grouped.rename(columns=column_mapping)

        # Add churn_rate column if we have churn data, otherwise use mock data
        if "churned" in grouped.columns:
            grouped["churn_rate"] = grouped["churned"]
        else:
            # Add mock churn rates for demo
            grouped["churn_rate"] = np.random.uniform(0.15, 0.30, len(grouped))

        # Ensure avg_revenue is present
        if "avg_revenue" not in grouped.columns:
            grouped["avg_revenue"] = np.random.uniform(2000, 10000, len(grouped))

        return grouped.to_dict(orient="records")
    except Exception as e:
        return {"error": f"Failed to process geographic data: {str(e)}"}


@router.get("/business-insights/revenue-at-risk")
async def get_revenue_at_risk():
    """Calculate revenue at risk based on churn predictions"""
    try:
        # Load customer data
        df = load_and_prep_data()
        if df is None:
            return {"error": "Could not load data"}

        # Get churn model
        churn_model = minio_client.get_model("logisticregression_classification")
        if churn_model is None:
            # Fallback to first available classification model
            models = [
                "randomforest_classification",
                "svm_classification",
                "decisiontree_classification",
            ]
            for model_name in models:
                churn_model = minio_client.get_model(model_name)
                if churn_model:
                    break

        if churn_model is None:
            # Fallback to simple rule-based churn prediction if no model is available
            print("Warning: No churn model found. Using rule-based fallback.")

            # Simple rule: if total_spent < average and days_since_last > average, then high churn prob
            avg_spent = df["total_sales"].mean()
            avg_days = (
                df["days_since_last_purchase"].mean()
                if "days_since_last_purchase" in df.columns
                else 100
            )

            probs = []
            for _, row in df.iterrows():
                prob = 0.2  # Base probability
                if row["total_sales"] < avg_spent:
                    prob += 0.3
                if (
                    "days_since_last_purchase" in row
                    and row["days_since_last_purchase"] > avg_days
                ):
                    prob += 0.3
                probs.append(min(0.95, prob))

            churn_probs = np.array(probs)

            # Skip model prediction part
            df["churn_probability"] = churn_probs
            df["revenue_at_risk"] = df["total_sales"] * churn_probs

            # Group by risk levels
            high_risk = df[df["churn_probability"] > 0.7]
            medium_risk = df[
                (df["churn_probability"] > 0.4) & (df["churn_probability"] <= 0.7)
            ]
            low_risk = df[df["churn_probability"] <= 0.4]

            return {
                "total_revenue_at_risk": float(df["revenue_at_risk"].sum()),
                "high_risk": {
                    "count": int(len(high_risk)),
                    "revenue_at_risk": float(high_risk["revenue_at_risk"].sum()),
                    "avg_churn_prob": (
                        float(high_risk["churn_probability"].mean())
                        if len(high_risk) > 0
                        else 0
                    ),
                },
                "medium_risk": {
                    "count": int(len(medium_risk)),
                    "revenue_at_risk": float(medium_risk["revenue_at_risk"].sum()),
                    "avg_churn_prob": (
                        float(medium_risk["churn_probability"].mean())
                        if len(medium_risk) > 0
                        else 0
                    ),
                },
                "low_risk": {
                    "count": int(len(low_risk)),
                    "revenue_at_risk": float(low_risk["revenue_at_risk"].sum()),
                    "avg_churn_prob": (
                        float(low_risk["churn_probability"].mean())
                        if len(low_risk) > 0
                        else 0
                    ),
                },
            }

        # Prepare features for prediction
        try:
            X = _prepare_features_for_inference(churn_model, df)
        except Exception as e:
            return {"error": f"Failed to prepare features: {str(e)}"}

        # Get churn probabilities
        if hasattr(churn_model, "predict_proba"):
            try:
                churn_probs = churn_model.predict_proba(X)[:, 1]
            except:
                # Fallback if predict_proba fails with prepared features
                churn_preds = churn_model.predict(X)
                churn_probs = churn_preds.astype(float)
        else:
            # Fallback to binary predictions
            churn_preds = churn_model.predict(X)
            churn_probs = churn_preds.astype(float)

        # Calculate revenue at risk
        df["churn_probability"] = churn_probs
        df["revenue_at_risk"] = df["total_sales"] * churn_probs

        # Group by risk levels
        high_risk = df[df["churn_probability"] > 0.7]
        medium_risk = df[
            (df["churn_probability"] > 0.4) & (df["churn_probability"] <= 0.7)
        ]
        low_risk = df[df["churn_probability"] <= 0.4]

        return {
            "total_revenue_at_risk": float(df["revenue_at_risk"].sum()),
            "high_risk": {
                "count": int(len(high_risk)),
                "revenue_at_risk": float(high_risk["revenue_at_risk"].sum()),
                "avg_churn_prob": (
                    float(high_risk["churn_probability"].mean())
                    if len(high_risk) > 0
                    else 0
                ),
            },
            "medium_risk": {
                "count": int(len(medium_risk)),
                "revenue_at_risk": float(medium_risk["revenue_at_risk"].sum()),
                "avg_churn_prob": (
                    float(medium_risk["churn_probability"].mean())
                    if len(medium_risk) > 0
                    else 0
                ),
            },
            "low_risk": {
                "count": int(len(low_risk)),
                "revenue_at_risk": float(low_risk["revenue_at_risk"].sum()),
                "avg_churn_prob": (
                    float(low_risk["churn_probability"].mean())
                    if len(low_risk) > 0
                    else 0
                ),
            },
        }
    except Exception as e:
        return {"error": f"Failed to calculate revenue at risk: {str(e)}"}


@router.post("/business-insights/campaign-roi")
async def calculate_campaign_roi(
    campaign_budget: float, target_segment: str = "high_risk"
):
    """Calculate ROI for a retention campaign"""
    try:
        # Get revenue at risk data
        revenue_data = await get_revenue_at_risk()
        if "error" in revenue_data:
            return revenue_data

        # Calculate potential savings based on target segment
        if target_segment == "high_risk":
            potential_revenue = revenue_data["high_risk"]["revenue_at_risk"]
            expected_retention_rate = 0.3  # 30% retention improvement
        elif target_segment == "medium_risk":
            potential_revenue = revenue_data["medium_risk"]["revenue_at_risk"]
            expected_retention_rate = 0.5  # 50% retention improvement
        else:  # all customers
            potential_revenue = revenue_data["total_revenue_at_risk"]
            expected_retention_rate = 0.2  # 20% retention improvement

        # Calculate ROI
        revenue_saved = potential_revenue * expected_retention_rate
        roi = (
            ((revenue_saved - campaign_budget) / campaign_budget) * 100
            if campaign_budget > 0
            else 0
        )

        return {
            "campaign_budget": campaign_budget,
            "target_segment": target_segment,
            "potential_revenue_at_risk": potential_revenue,
            "expected_retention_improvement": expected_retention_rate * 100,
            "revenue_saved": revenue_saved,
            "roi_percentage": roi,
            "payback_period_months": (
                (campaign_budget / (revenue_saved / 12)) if revenue_saved > 0 else None
            ),
            "recommendation": (
                "Proceed"
                if roi > 50
                else "Consider alternative strategies" if roi > 0 else "Not recommended"
            ),
        }
    except Exception as e:
        return {"error": f"Failed to calculate campaign ROI: {str(e)}"}


@router.get("/business-insights/customer-segments")
async def get_customer_segments():
    """Get customer segmentation data"""
    try:
        df = load_and_prep_data()
        if df is None:
            return {"error": "Could not load data"}

        # Get churn model for predictions
        churn_model = minio_client.get_model("logisticregression_classification")
        if churn_model is None:
            return {"error": "No classification model available"}

        # Prepare features - use the same approach as revenue-at-risk
        try:
            X = _prepare_features_for_inference(churn_model, df)
        except Exception as e:
            return {"error": f"Failed to prepare features: {str(e)}"}

        # Get churn probabilities
        if hasattr(churn_model, "predict_proba"):
            try:
                churn_probs = churn_model.predict_proba(X)[:, 1]
            except:
                churn_preds = churn_model.predict(X)
                churn_probs = churn_preds.astype(float)
        else:
            churn_preds = churn_model.predict(X)
            churn_probs = churn_preds.astype(float)

        df["churn_probability"] = churn_probs

        # Create segments based on value and risk
        # High value: top 25% of total_sales
        high_value_threshold = df["total_sales"].quantile(0.75)

        # Create segments
        segments = []

        # Champions: High value, Low risk
        champions = df[
            (df["total_sales"] >= high_value_threshold)
            & (df["churn_probability"] <= 0.3)
        ]
        segments.append(
            {
                "segment": "Champions",
                "count": int(len(champions)),
                "avg_total_sales": (
                    float(champions["total_sales"].mean()) if len(champions) > 0 else 0
                ),
                "avg_churn_prob": (
                    float(champions["churn_probability"].mean())
                    if len(champions) > 0
                    else 0
                ),
                "total_revenue": float(champions["total_sales"].sum()),
                "description": "High value customers with low churn risk",
            }
        )

        # Loyal Customers: High value, Medium risk
        loyal = df[
            (df["total_sales"] >= high_value_threshold)
            & (df["churn_probability"] > 0.3)
            & (df["churn_probability"] <= 0.7)
        ]
        segments.append(
            {
                "segment": "Loyal Customers",
                "count": int(len(loyal)),
                "avg_total_sales": (
                    float(loyal["total_sales"].mean()) if len(loyal) > 0 else 0
                ),
                "avg_churn_prob": (
                    float(loyal["churn_probability"].mean()) if len(loyal) > 0 else 0
                ),
                "total_revenue": float(loyal["total_sales"].sum()),
                "description": "High value customers with medium churn risk",
            }
        )

        # At Risk: High value, High risk
        at_risk = df[
            (df["total_sales"] >= high_value_threshold)
            & (df["churn_probability"] > 0.7)
        ]
        segments.append(
            {
                "segment": "At Risk",
                "count": int(len(at_risk)),
                "avg_total_sales": (
                    float(at_risk["total_sales"].mean()) if len(at_risk) > 0 else 0
                ),
                "avg_churn_prob": (
                    float(at_risk["churn_probability"].mean())
                    if len(at_risk) > 0
                    else 0
                ),
                "total_revenue": float(at_risk["total_sales"].sum()),
                "description": "High value customers with high churn risk",
            }
        )

        # New Customers: Low value, Low risk (assuming recent customers have lower total sales)
        new_customers = df[
            (df["total_sales"] < high_value_threshold)
            & (df["churn_probability"] <= 0.3)
        ]
        segments.append(
            {
                "segment": "New Customers",
                "count": int(len(new_customers)),
                "avg_total_sales": (
                    float(new_customers["total_sales"].mean())
                    if len(new_customers) > 0
                    else 0
                ),
                "avg_churn_prob": (
                    float(new_customers["churn_probability"].mean())
                    if len(new_customers) > 0
                    else 0
                ),
                "total_revenue": float(new_customers["total_sales"].sum()),
                "description": "Lower value customers with low churn risk",
            }
        )

        # Need Attention: Low value, High risk
        need_attention = df[
            (df["total_sales"] < high_value_threshold) & (df["churn_probability"] > 0.7)
        ]
        segments.append(
            {
                "segment": "Need Attention",
                "count": int(len(need_attention)),
                "avg_total_sales": (
                    float(need_attention["total_sales"].mean())
                    if len(need_attention) > 0
                    else 0
                ),
                "avg_churn_prob": (
                    float(need_attention["churn_probability"].mean())
                    if len(need_attention) > 0
                    else 0
                ),
                "total_revenue": float(need_attention["total_sales"].sum()),
                "description": "Lower value customers with high churn risk",
            }
        )

        return {
            "segments": segments,
            "total_customers": int(len(df)),
            "total_revenue": float(df["total_sales"].sum()),
        }
    except Exception as e:
        return {"error": f"Failed to get customer segments: {str(e)}"}


@router.get("/export/customer-segments")
async def export_customer_segments(format: str = "csv"):
    """Export customer segmentation data"""
    try:
        # Get customer segments data
        segments_data = await get_customer_segments()
        if "error" in segments_data:
            return segments_data

        # Convert to DataFrame
        segments_df = pd.DataFrame(segments_data["segments"])

        # Add metadata
        segments_df["export_date"] = pd.Timestamp.now()
        segments_df["total_customers"] = segments_data["total_customers"]
        segments_df["total_revenue"] = segments_data["total_revenue"]

        if format.lower() == "csv":
            # Convert to CSV
            csv_data = segments_df.to_csv(index=False)
            from fastapi.responses import Response

            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=customer_segments.csv"
                },
            )
        elif format.lower() == "json":
            return segments_data
        else:
            return {"error": "Unsupported format. Use 'csv' or 'json'"}

    except Exception as e:
        return {"error": f"Failed to export customer segments: {str(e)}"}


def _traverse_tree_for_features(node, feature_usage, feature_cols):
    """Helper function to traverse tree and collect feature usage"""
    if hasattr(node, "feature") and node.feature is not None:
        if node.feature < len(feature_cols):
            feature_name = feature_cols[node.feature]
            feature_usage[feature_name] = feature_usage.get(feature_name, 0) + 1

    if hasattr(node, "left") and node.left is not None:
        _traverse_tree_for_features(node.left, feature_usage, feature_cols)
    if hasattr(node, "right") and node.right is not None:
        _traverse_tree_for_features(node.right, feature_usage, feature_cols)


@router.get("/model-explainability/feature-importance/{model_name}")
async def get_feature_importance(model_name: str):
    """Get feature importance for a specific model"""
    try:
        model = minio_client.get_model(model_name)
        if model is None:
            return {"error": f"Model {model_name} not found"}

        # Get training data to extract feature names
        df = load_and_prep_data()
        if df is None:
            return {"error": "Could not load training data"}

        # Get feature names from model if available, otherwise from data
        if hasattr(model, "feature_names"):
            feature_cols = model.feature_names
        else:
            # Fallback to data columns
            drop_cols = ["total_sales", "churned", "clv_per_year"]
            feature_cols = [c for c in df.columns if c not in drop_cols]

        print(f"Using feature names: {feature_cols}")
        print(f"Number of feature columns: {len(feature_cols)}")
        print(f"Model has weights: {hasattr(model, 'weights')}")
        if hasattr(model, "weights"):
            print(
                f"Model weights shape: {model.weights.shape if hasattr(model.weights, 'shape') else 'no shape'}"
            )
            print(
                f"Model weights length: {len(model.weights) if hasattr(model.weights, '__len__') else 'no len'}"
            )

        # Get feature importance based on model type
        importance_data = []

        # Debug: Check model attributes
        print(
            f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}"
        )
        print(f"Has feature_importances_: {hasattr(model, 'feature_importances_')}")
        print(f"Has coef_: {hasattr(model, 'coef_')}")
        print(f"Has weights: {hasattr(model, 'weights')}")

        if hasattr(model, "feature_importances_"):
            # Tree-based models (Random Forest, XGBoost, etc.)
            importances = model.feature_importances_
            for i, (feature, importance) in enumerate(zip(feature_cols, importances)):
                importance_data.append(
                    {"feature": feature, "importance": float(importance), "rank": i + 1}
                )
        elif hasattr(model, "coef_"):
            # Linear models (Logistic Regression, Linear Regression, etc.)
            coefs = (
                np.abs(model.coef_[0])
                if len(model.coef_.shape) > 1
                else np.abs(model.coef_)
            )
            for i, (feature, coef) in enumerate(zip(feature_cols, coefs)):
                importance_data.append(
                    {"feature": feature, "importance": float(coef), "rank": i + 1}
                )
        elif hasattr(model, "weights"):
            # Custom models with weights attribute
            print(f"Model weights type: {type(model.weights)}")
            print(f"Model weights value: {model.weights}")

            try:
                weights = (
                    np.abs(model.weights.flatten())
                    if hasattr(model.weights, "flatten")
                    else np.abs(model.weights)
                )
                # Ensure we have the right number of weights
                if len(weights) == len(feature_cols):
                    for i, (feature, weight) in enumerate(zip(feature_cols, weights)):
                        importance_data.append(
                            {
                                "feature": feature,
                                "importance": float(weight),
                                "rank": i + 1,
                            }
                        )
                else:
                    # Weight count mismatch - skip this model
                    pass
            except Exception as e:
                print(f"Error processing weights: {e}")
                # For tree-based models, try to get feature importance from individual trees
                if hasattr(model, "trees"):
                    # This is a simplified approach - sum up feature usage across all trees
                    feature_usage = {}
                    for i, tree in enumerate(model.trees):
                        if hasattr(tree, "feature_importance"):
                            for j, importance in enumerate(tree.feature_importance):
                                if j < len(feature_cols):
                                    feature_name = feature_cols[j]
                                    feature_usage[feature_name] = (
                                        feature_usage.get(feature_name, 0) + importance
                                    )
                        elif hasattr(tree, "feature_importances_"):
                            for j, importance in enumerate(tree.feature_importances_):
                                if j < len(feature_cols):
                                    feature_name = feature_cols[j]
                                    feature_usage[feature_name] = (
                                        feature_usage.get(feature_name, 0) + importance
                                    )
                        elif hasattr(tree, "root"):
                            # Try to traverse the tree to get feature usage
                            _traverse_tree_for_features(
                                tree.root, feature_usage, feature_cols
                            )

                    # Convert to list format
                    for i, (feature, importance) in enumerate(feature_usage.items()):
                        importance_data.append(
                            {
                                "feature": feature,
                                "importance": float(importance),
                                "rank": i + 1,
                            }
                        )
                else:
                    print("No trees attribute found")
        else:
            # For models without built-in feature importance, use permutation importance
            # This is a simplified version - in production, you'd want to implement proper permutation importance
            return {"error": "Feature importance not available for this model type"}

        # Sort by importance
        importance_data.sort(key=lambda x: x["importance"], reverse=True)

        # Update ranks
        for i, item in enumerate(importance_data):
            item["rank"] = i + 1

        return {
            "model_name": model_name,
            "feature_importance": importance_data,
            "total_features": len(importance_data),
        }
    except Exception as e:
        return {"error": f"Failed to get feature importance: {str(e)}"}


@router.post("/model-explainability/shap-values/{model_name}")
async def get_shap_values(model_name: str, sample_data: Dict[str, Any]):
    """Get SHAP values for a specific prediction"""
    try:
        if not SHAP_AVAILABLE:
            return {
                "error": "SHAP library not available. Install with: pip install shap"
            }

        model = minio_client.get_model(model_name)
        if model is None:
            return {"error": f"Model {model_name} not found"}

        # Get training data
        df = load_and_prep_data()
        if df is None:
            return {"error": "Could not load training data"}

        # Prepare features
        drop_cols = ["total_sales", "churned", "clv_per_year"]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        # Convert input to DataFrame
        input_df = pd.DataFrame([sample_data])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)
        input_df = input_df.astype(float)

        # Create SHAP explainer
        if hasattr(model, "predict_proba"):
            # Classification model
            explainer = (
                shap.TreeExplainer(model)
                if hasattr(model, "feature_importances_")
                else shap.LinearExplainer(model, input_df)
            )
            shap_values = explainer.shap_values(input_df)

            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Get prediction probability
            prediction_proba = model.predict_proba(input_df)[0][1]

            return {
                "model_name": model_name,
                "prediction_probability": float(prediction_proba),
                "shap_values": [
                    {
                        "feature": feature,
                        "value": float(shap_val),
                        "impact": "positive" if shap_val > 0 else "negative",
                    }
                    for feature, shap_val in zip(feature_cols, shap_values[0])
                ],
                "expected_value": (
                    float(explainer.expected_value[1])
                    if hasattr(explainer, "expected_value")
                    else None
                ),
            }
        else:
            # Regression model
            explainer = (
                shap.TreeExplainer(model)
                if hasattr(model, "feature_importances_")
                else shap.LinearExplainer(model, input_df)
            )
            shap_values = explainer.shap_values(input_df)

            # Get prediction
            prediction = model.predict(input_df)[0]

            return {
                "model_name": model_name,
                "prediction": float(prediction),
                "shap_values": [
                    {
                        "feature": feature,
                        "value": float(shap_val),
                        "impact": "positive" if shap_val > 0 else "negative",
                    }
                    for feature, shap_val in zip(feature_cols, shap_values[0])
                ],
                "expected_value": (
                    float(explainer.expected_value)
                    if hasattr(explainer, "expected_value")
                    else None
                ),
            }

    except Exception as e:
        return {"error": f"Failed to get SHAP values: {str(e)}"}


@router.get("/business-insights/clv-trends")
async def get_clv_trends(period: str = "monthly"):
    """Get Customer Lifetime Value trends over time"""
    try:
        # Load customer data
        df = load_and_prep_data()
        if df is None:
            return {"error": "Could not load data"}

        # Calculate CLV from available data since we don't have a direct CLV column
        # CLV can be estimated as: (Average Purchase Value × Purchase Frequency × Customer Lifespan)
        # We'll use available metrics to create a reasonable CLV estimate

        # Get churn model for predictions
        churn_model = minio_client.get_model("logisticregression_classification")
        if churn_model is None:
            return {"error": "Churn model not available"}

        print(f"Model loaded: {churn_model}")
        print(f"Model has n_features_in_: {hasattr(churn_model, 'n_features_in_')}")
        if hasattr(churn_model, "n_features_in_"):
            print(f"Expected features: {churn_model.n_features_in_}")

        # Check what attributes the model has
        print(
            f"Model attributes: {[attr for attr in dir(churn_model) if not attr.startswith('_') and 'feature' in attr.lower()]}"
        )

        # Try to get expected feature count from model
        n_features = 43  # Default based on the error message
        if hasattr(churn_model, "feature_names"):
            feature_names = getattr(churn_model, "feature_names")
            print(f"Feature names: {len(feature_names) if feature_names else 'None'}")
            if feature_names and len(feature_names) > 0:
                n_features = len(feature_names)

        print(f"Using expected features: {n_features}")

        # Prepare features for churn prediction
        # Use the same logic as the predict endpoint to handle feature count mismatch
        drop_cols = ["total_sales", "churned"]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        # Handle missing or non-numeric columns
        X = df[feature_cols].copy()
        for col in X.columns:
            if X[col].dtype == "object":
                try:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                except:
                    X[col] = 0
        X = X.fillna(0)

        # Handle feature count mismatch (model expects 43, we have 121)
        # Use the same logic as the predict endpoint
        X_numeric = X.values
        print(f"Original X shape: {X_numeric.shape}")

        current_features = X_numeric.shape[1]
        print(f"Current features: {current_features}, Expected: {n_features}")

        if current_features < n_features:
            # Pad with zeros to match expected feature count
            padding = np.zeros((X_numeric.shape[0], n_features - current_features))
            X_numeric = np.hstack([X_numeric, padding])
            print(f"Padded X shape: {X_numeric.shape}")
        elif current_features > n_features:
            # If we have more features than expected, truncate
            X_numeric = X_numeric[:, :n_features]
            print(f"Truncated X shape: {X_numeric.shape}")

        # Get churn probabilities
        if hasattr(churn_model, "predict_proba"):
            churn_probs = churn_model.predict_proba(X_numeric)[:, 1]
        else:
            churn_preds = churn_model.predict(X_numeric)
            churn_probs = churn_preds.astype(float)

        df["churn_probability"] = churn_probs

        # Calculate estimated CLV from available data
        # CLV = Average Purchase Value × Purchase Frequency × Customer Lifespan
        # We'll estimate customer lifespan based on membership years and recency

        # Estimate customer lifespan (in years) based on available data
        # Use membership years as a proxy for customer relationship duration
        df["estimated_lifespan"] = (
            df["membership_years"] + 1
        )  # Add 1 to account for current year

        # Adjust lifespan based on recency (days since last purchase)
        # More recent purchases suggest longer relationship
        max_recency = (
            df["days_since_last_purchase"].max()
            if "days_since_last_purchase" in df.columns
            else 365
        )
        df["recency_factor"] = (
            1 - (df["days_since_last_purchase"] / max_recency)
            if "days_since_last_purchase" in df.columns
            else 0.5
        )
        df["adjusted_lifespan"] = df["estimated_lifespan"] * (
            1 + df["recency_factor"] * 0.3
        )

        # Calculate CLV: Average Purchase Value × Purchase Frequency × Adjusted Lifespan
        # Use avg_purchase_value and estimate frequency from total_transactions and membership_years
        df["estimated_annual_frequency"] = df["total_transactions"] / df[
            "membership_years"
        ].replace(0, 1)
        df["clv_estimate"] = (
            df["avg_purchase_value"]
            * df["estimated_annual_frequency"]
            * df["adjusted_lifespan"]
        )

        # Calculate expected CLV (adjust based on churn probability)
        # For customers with higher churn probability, reduce expected CLV
        df["expected_clv"] = df["clv_estimate"] * (1 - df["churn_probability"])

        # Create time-based trends
        # Use transaction_date if available, otherwise create a mock date
        if "transaction_date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["transaction_date"])
            except:
                # If date conversion fails, create mock dates
                df["date"] = (
                    pd.Timestamp.now()
                    - pd.Timedelta(days=365)
                    + pd.to_timedelta(df.index, unit="D")
                )
        else:
            # Create mock dates for demonstration
            df["date"] = (
                pd.Timestamp.now()
                - pd.Timedelta(days=365)
                + pd.to_timedelta(df.index, unit="D")
            )

        # Group by time period
        if period == "monthly":
            df["period"] = df["date"].dt.to_period("M")
        elif period == "quarterly":
            df["period"] = df["date"].dt.to_period("Q")
        elif period == "yearly":
            df["period"] = df["date"].dt.to_period("Y")
        else:
            df["period"] = df["date"].dt.to_period("M")  # default to monthly

        # Calculate trends by period
        trends = []
        for period_name, group in df.groupby("period"):
            trends.append(
                {
                    "period": str(period_name),
                    "avg_clv": float(group["clv_estimate"].mean()),
                    "avg_expected_clv": float(group["expected_clv"].mean()),
                    "avg_churn_probability": float(group["churn_probability"].mean()),
                    "customer_count": int(len(group)),
                    "total_revenue": float(group["total_sales"].sum()),
                    "high_value_customers": int(
                        len(
                            group[
                                group["clv_estimate"]
                                > group["clv_estimate"].quantile(0.75)
                            ]
                        )
                    ),
                    "at_risk_customers": int(
                        len(group[group["churn_probability"] > 0.7])
                    ),
                }
            )

        # Sort by period
        trends.sort(key=lambda x: x["period"])

        # Calculate growth rates
        if len(trends) > 1:
            for i in range(1, len(trends)):
                current = trends[i]
                previous = trends[i - 1]

                if previous["avg_clv"] > 0:
                    current["clv_growth_rate"] = float(
                        (current["avg_clv"] - previous["avg_clv"]) / previous["avg_clv"]
                    )
                else:
                    current["clv_growth_rate"] = 0.0

                if previous["avg_expected_clv"] > 0:
                    current["expected_clv_growth_rate"] = float(
                        (current["avg_expected_clv"] - previous["avg_expected_clv"])
                        / previous["avg_expected_clv"]
                    )
                else:
                    current["expected_clv_growth_rate"] = 0.0

        # Calculate insights
        avg_clv = df["clv_estimate"].mean()
        avg_expected_clv = df["expected_clv"].mean()
        avg_churn_prob = df["churn_probability"].mean()

        insights = {
            "current_avg_clv": float(avg_clv),
            "current_avg_expected_clv": float(avg_expected_clv),
            "current_avg_churn_probability": float(avg_churn_prob),
            "total_customers": int(len(df)),
            "total_revenue": float(df["total_sales"].sum()),
            "clv_revenue_ratio": (
                float(avg_clv / (df["total_sales"].mean() / len(df)))
                if len(df) > 0
                else 0
            ),
            "high_value_segment_size": int(
                len(df[df["clv_estimate"] > df["clv_estimate"].quantile(0.75)])
            ),
            "at_risk_segment_size": int(len(df[df["churn_probability"] > 0.7])),
            "period": period,
            "trend_periods": len(trends),
        }

        return {"trends": trends, "insights": insights, "period": period}

    except Exception as e:
        return {"error": f"Failed to get CLV trends: {str(e)}"}
