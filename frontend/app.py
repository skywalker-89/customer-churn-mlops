import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Churn MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_URL = "http://localhost:8000/api"

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.1rem;
    }
    .scratch-badge {
        background-color: #ff7f0e;
        color: white;
    }
    .sklearn-badge {
        background-color: #2ca02c;
        color: white;
    }

</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-header">🎯 Customer Churn MLOps Dashboard</div>',
    unsafe_allow_html=True,
)

# Sidebar for navigation
st.sidebar.title("🚀 Dashboard Navigation")
dashboard_section = st.sidebar.radio(
    "Choose a section:",
    [
        "📊 Model Performance",
        "🔮 Prediction Interface",
        "🗺️ Geographic Visualization",
        "📈 Business Insights",
    ],
)


# Cache data fetching functions
@st.cache_data(ttl=300)
def fetch_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"classification": [], "regression": []}


@st.cache_data(ttl=300)
def fetch_features():
    try:
        response = requests.get(f"{API_URL}/features")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


@st.cache_data(ttl=300)
def fetch_geo_data():
    try:
        response = requests.get(f"{API_URL}/customers/geo")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


@st.cache_data(ttl=300)
def fetch_revenue_at_risk():
    try:
        response = requests.get(f"{API_URL}/business-insights/revenue-at-risk")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


@st.cache_data(ttl=300)
def fetch_customer_segments():
    try:
        response = requests.get(f"{API_URL}/business-insights/customer-segments")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


@st.cache_data(ttl=300)
def fetch_location_options():
    """Fetch valid location options from the backend"""
    try:
        response = requests.get(f"{API_URL}/features")
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])

            # Extract valid store locations
            store_locations = []
            for feature in features:
                if feature.startswith("store_location_"):
                    location = feature.replace("store_location_", "")
                    store_locations.append(location)

            # Extract valid customer cities
            customer_cities = []
            for feature in features:
                if feature.startswith("customer_city_"):
                    city = feature.replace("customer_city_", "")
                    customer_cities.append(city)

            # Extract valid store cities
            store_cities = []
            for feature in features:
                if feature.startswith("store_city_"):
                    city = feature.replace("store_city_", "")
                    store_cities.append(city)

            return {
                "store_locations": sorted(list(set(store_locations))),
                "customer_cities": sorted(list(set(customer_cities))),
                "store_cities": sorted(list(set(store_cities))),
            }
    except Exception as e:
        print(f"Error fetching location options: {e}")
        pass

    # Return default options if API fails
    return {
        "store_locations": ["Location A", "Location B", "Location C", "Location D"],
        "customer_cities": ["City A", "City B", "City C", "City D"],
        "store_cities": ["City A", "City B", "City C", "City D"],
    }


@st.cache_data(ttl=300)
def fetch_feature_importance(model_name):
    try:
        response = requests.get(
            f"{API_URL}/model-explainability/feature-importance/{model_name}"
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


@st.cache_data(ttl=10)
def fetch_available_models():
    """Fetch available models from the backend"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
    except:
        pass

    # Fallback
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


def render_prediction_form(key_prefix, location_options):
    """Render the input form for prediction and return the input values"""
    # Basic customer info
    st.subheader("👤 Basic Information")
    age = st.slider("Age", 18, 100, 35, key=f"{key_prefix}_age")
    income = st.number_input(
        "Annual Income ($)", 0, 200000, 50000, step=1000, key=f"{key_prefix}_income"
    )

    # Geographic info
    st.subheader("📍 Location")

    # Store location
    store_location = st.selectbox(
        "Store Location",
        location_options["store_locations"],
        key=f"{key_prefix}_store_location",
    )

    # Customer city
    customer_city = st.selectbox(
        "Customer City",
        location_options["customer_cities"],
        key=f"{key_prefix}_customer_city",
    )

    # Store city
    store_city = st.selectbox(
        "Store City", location_options["store_cities"], key=f"{key_prefix}_store_city"
    )

    # Purchase behavior
    st.subheader("🛒 Purchase Behavior")

    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input(
            "Total Quantity of Items",
            1,
            1000,
            10,
            step=1,
            key=f"{key_prefix}_quantity",
        )
    with col2:
        unit_price = st.number_input(
            "Average Unit Price ($)",
            1.0,
            5000.0,
            100.0,
            step=10.0,
            key=f"{key_prefix}_unit_price",
        )

    discount_applied = st.slider(
        "Discount Applied (%)",
        0.0,
        1.0,
        0.0,
        step=0.05,
        format="%.2f",
        key=f"{key_prefix}_discount_applied",
    )

    # Calculate derived values for display
    total_spent_calculated = quantity * unit_price * (1 - discount_applied)
    st.info(
        f"💰 Calculated Total Spent: ${total_spent_calculated:,.2f} (Quantity × Unit Price × (1 - Discount))"
    )

    # Hidden total_spent field (we use calculated value but keep API compatibility)
    total_spent = total_spent_calculated

    num_purchases = st.slider(
        "Number of Purchases", 1, 100, 10, key=f"{key_prefix}_num_purchases"
    )

    # Detailed purchase breakdown for feature engineering
    online_purchases = st.slider(
        "Online Purchases",
        0,
        num_purchases,
        int(num_purchases * 0.4),
        key=f"{key_prefix}_online_purchases",
    )
    in_store_purchases = num_purchases - online_purchases
    st.caption(f"In-Store Purchases: {in_store_purchases}")

    avg_purchase = total_spent / max(num_purchases, 1)
    st.metric("Average Purchase Amount", f"${avg_purchase:.2f}")

    # Customer engagement
    st.subheader("📱 Engagement")

    col1, col2 = st.columns(2)
    with col1:
        app_usage_frequency = st.selectbox(
            "App Usage Frequency",
            ["Low", "Medium", "High"],
            index=1,
            key=f"{key_prefix}_app_usage_frequency",
            help="Frequency of mobile app usage",
        )
    with col2:
        social_media_engagement = st.selectbox(
            "Social Media Engagement",
            ["Low", "Medium", "High"],
            index=0,
            key=f"{key_prefix}_social_media_engagement",
            help="Level of interaction on social media",
        )

    days_since_last_purchase = st.slider(
        "Days Since Last Purchase",
        0,
        365,
        30,
        key=f"{key_prefix}_days_since_last_purchase",
    )
    website_visits = st.slider(
        "Website Visits (last month)", 0, 50, 5, key=f"{key_prefix}_website_visits"
    )

    # Additional features
    st.subheader("⚙️ Additional Features")
    distance_to_store = st.slider(
        "Distance to Store (km)",
        0.0,
        50.0,
        5.0,
        step=0.5,
        key=f"{key_prefix}_distance_to_store",
    )

    col1, col2 = st.columns(2)
    with col1:
        has_complained = st.checkbox(
            "Has Filed Complaint", key=f"{key_prefix}_has_complained"
        )
        is_active_member = st.checkbox(
            "Is Active Member", value=True, key=f"{key_prefix}_is_active_member"
        )
    with col2:
        email_subscriptions = st.checkbox(
            "Email Subscriber", value=True, key=f"{key_prefix}_email_subscriptions"
        )
        has_returned_items = st.checkbox(
            "Has Returned Items", key=f"{key_prefix}_has_returned_items"
        )

    product_return_rate = 0.0
    if has_returned_items:
        product_return_rate = st.slider(
            "Return Rate", 0.0, 1.0, 0.1, key=f"{key_prefix}_product_return_rate"
        )

    return {
        "age": age,
        "income": income,
        "store_location": store_location,
        "customer_city": customer_city,
        "store_city": store_city,
        "total_spent": total_spent,
        "num_purchases": num_purchases,
        "online_purchases": online_purchases,
        "in_store_purchases": in_store_purchases,
        "avg_purchase": avg_purchase,
        "days_since_last_purchase": days_since_last_purchase,
        "website_visits": website_visits,
        "has_complained": has_complained,
        "is_active_member": is_active_member,
        "email_subscriptions": 1 if email_subscriptions else 0,
        "distance_to_store": distance_to_store,
        "product_return_rate": product_return_rate,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_applied": discount_applied,
        "app_usage_frequency": app_usage_frequency,
        "social_media_engagement": social_media_engagement,
    }


# Model Performance Dashboard
if dashboard_section == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")

    # Fetch metrics data
    metrics_data = fetch_metrics()

    if not metrics_data.get("classification") and not metrics_data.get("regression"):
        st.warning(
            "⚠️ No model metrics available. Please check if the backend is running and models have been trained."
        )

        # Show cached demo data if no real data
        st.info("📊 Showing demo data for demonstration purposes:")

        # Demo classification models
        demo_classification = [
            {
                "model": "Logistic Regression (Scratch)",
                "accuracy": 0.834,
                "f1_score": 0.676,
                "precision": 0.629,
                "recall": 0.730,
            },
            {
                "model": "Random Forest (Sklearn)",
                "accuracy": 0.844,
                "f1_score": 0.649,
                "precision": 0.696,
                "recall": 0.607,
            },
            {
                "model": "SVM (Scratch)",
                "accuracy": 0.759,
                "f1_score": 0.640,
                "precision": 0.495,
                "recall": 0.906,
            },
            {
                "model": "Decision Tree (Scratch)",
                "accuracy": 0.841,
                "f1_score": 0.639,
                "precision": 0.694,
                "recall": 0.592,
            },
        ]

        demo_regression = [
            {
                "model": "Linear Regression (Scratch)",
                "r2": 0.752,
                "rmse": 0.487,
                "mae": 0.392,
            },
            {
                "model": "Random Forest Regressor (Sklearn)",
                "r2": 0.823,
                "rmse": 0.412,
                "mae": 0.334,
            },
            {"model": "XGBoost Regressor", "r2": 0.845, "rmse": 0.398, "mae": 0.321},
        ]

        metrics_data = {
            "classification": demo_classification,
            "regression": demo_regression,
        }

    # Classification Models Section
    if metrics_data.get("classification"):
        st.header("🎯 Classification Models")

        # Display Static Visualization if available
        if metrics_data.get("visualizations") and metrics_data["visualizations"].get(
            "classification"
        ):
            st.subheader("📊 Static Performance Visualization")
            image_path = metrics_data["visualizations"]["classification"]
            # Ensure full URL
            if image_path.startswith("/"):
                full_image_url = f"http://localhost:8000{image_path}"
            else:
                full_image_url = image_path

            st.image(
                full_image_url,
                caption="Classification Model Performance",
                use_container_width=True,
            )

        classification_df = pd.DataFrame(metrics_data["classification"])

        # Model selection for detailed comparison
        selected_models = st.multiselect(
            "Select models to compare:",
            classification_df["model"].tolist(),
            default=classification_df["model"].tolist()[:3],
        )

        if selected_models:
            filtered_df = classification_df[
                classification_df["model"].isin(selected_models)
            ]

            # Performance metrics visualization
            col1, col2 = st.columns(2)

            with col1:
                # Accuracy comparison
                fig_accuracy = px.bar(
                    filtered_df,
                    x="model",
                    y="accuracy",
                    title="📊 Model Accuracy Comparison",
                    color="accuracy",
                    color_continuous_scale="Blues",
                    height=400,
                )
                fig_accuracy.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_accuracy, width="stretch")

                # F1-Score comparison
                fig_f1 = px.bar(
                    filtered_df,
                    x="model",
                    y="f1_score",
                    title="🎯 F1-Score Comparison",
                    color="f1_score",
                    color_continuous_scale="Greens",
                    height=400,
                )
                fig_f1.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_f1, width="stretch")

            with col2:
                # Precision comparison
                fig_precision = px.bar(
                    filtered_df,
                    x="model",
                    y="precision",
                    title="🎯 Precision Comparison",
                    color="precision",
                    color_continuous_scale="Oranges",
                    height=400,
                )
                fig_precision.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_precision, width="stretch")

                # Recall comparison
                fig_recall = px.bar(
                    filtered_df,
                    x="model",
                    y="recall",
                    title="🎯 Recall Comparison",
                    color="recall",
                    color_continuous_scale="Reds",
                    height=400,
                )
                fig_recall.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_recall, width="stretch")

            # Detailed metrics table
            st.subheader("📋 Detailed Classification Metrics")

            # Format the dataframe for better display
            display_df = filtered_df.copy()
            for col in ["accuracy", "f1_score", "precision", "recall"]:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                )

            st.dataframe(display_df, width="stretch")

            # Model badges
            st.subheader("🏷️ Model Implementation Types")
            col1, col2 = st.columns(2)

            with col1:
                scratch_models = [
                    model for model in selected_models if "(Scratch)" in model
                ]
                if scratch_models:
                    st.markdown("**From Scratch Models:**")
                    for model in scratch_models:
                        st.markdown(
                            f'<span class="model-badge scratch-badge">{model}</span>',
                            unsafe_allow_html=True,
                        )

            with col2:
                sklearn_models = [
                    model for model in selected_models if "(Sklearn)" in model
                ]
                if sklearn_models:
                    st.markdown("**Scikit-learn Models:**")
                    for model in sklearn_models:
                        st.markdown(
                            f'<span class="model-badge sklearn-badge">{model}</span>',
                            unsafe_allow_html=True,
                        )

    # Regression Models Section
    if metrics_data.get("regression"):
        st.header("📈 Regression Models")

        # Display Static Visualization if available
        if metrics_data.get("visualizations") and metrics_data["visualizations"].get(
            "regression"
        ):
            st.subheader("📊 Static Performance Visualization")
            image_path = metrics_data["visualizations"]["regression"]
            # Ensure full URL
            if image_path.startswith("/"):
                full_image_url = f"http://localhost:8000{image_path}"
            else:
                full_image_url = image_path

            st.image(
                full_image_url,
                caption="Regression Model Performance",
                use_container_width=True,
            )

        regression_df = pd.DataFrame(metrics_data["regression"])

        # Model selection for regression
        selected_regression_models = st.multiselect(
            "Select regression models to compare:",
            regression_df["model"].tolist(),
            default=regression_df["model"].tolist()[:2],
        )

        if selected_regression_models:
            filtered_regression_df = regression_df[
                regression_df["model"].isin(selected_regression_models)
            ]

            col1, col2, col3 = st.columns(3)

            with col1:
                # R² comparison
                fig_r2 = px.bar(
                    filtered_regression_df,
                    x="model",
                    y="r2",
                    title="📊 R² Score Comparison",
                    color="r2",
                    color_continuous_scale="Purples",
                    height=400,
                )
                fig_r2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_r2, width="stretch")

            with col2:
                # RMSE comparison
                fig_rmse = px.bar(
                    filtered_regression_df,
                    x="model",
                    y="rmse",
                    title="📊 RMSE Comparison (Lower is Better)",
                    color="rmse",
                    color_continuous_scale="Reds",
                    height=400,
                )
                fig_rmse.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_rmse, width="stretch")

            with col3:
                # MAE comparison
                fig_mae = px.bar(
                    filtered_regression_df,
                    x="model",
                    y="mae",
                    title="📊 MAE Comparison (Lower is Better)",
                    color="mae",
                    color_continuous_scale="Blues",
                    height=400,
                )
                fig_mae.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_mae, width="stretch")

            # Detailed regression metrics table
            st.subheader("📋 Detailed Regression Metrics")

            # Format the dataframe for better display
            display_regression_df = filtered_regression_df.copy()
            for col in ["r2", "rmse", "mae"]:
                display_regression_df[col] = display_regression_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                )

            st.dataframe(display_regression_df, width="stretch")

# Prediction Interface
elif dashboard_section == "🔮 Prediction Interface":
    st.title("🔮 Prediction Interface")

    # Fetch features data
    features_data = fetch_features()

    if not features_data:
        st.warning(
            "⚠️ No feature information available. Please check if the backend is running."
        )
    else:
        st.success("✅ Feature data loaded successfully!")

        # Fetch available models and location options
        available_models = fetch_available_models()
        location_options = fetch_location_options()

        # Create tabs for Classification and Regression
        # Use radio button for robust state management during form submissions
        prediction_type = st.radio(
            "Select Prediction Type",
            ["Churn Prediction (Classification)", "CLV Prediction (Regression)"],
            horizontal=True,
            label_visibility="collapsed",
            key="prediction_type_selector",
        )

        st.markdown("---")

        # --- Classification Tab ---
        if prediction_type == "Churn Prediction (Classification)":
            st.header("🤔 Churn Prediction")

            # Create two columns for input and results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🤷‍♂️ Customer Input")
                with st.form("classification_form"):
                    # Model Selection
                    class_models = available_models.get("classification", [])
                    if not class_models:
                        class_models = ["logistic_regression_classification"]

                    selected_class_model = st.selectbox(
                        "Select Classification Model",
                        class_models,
                        index=0,
                        key="class_model_select",
                    )

                    input_values = render_prediction_form("class", location_options)
                    submitted_class = st.form_submit_button(
                        "🚀 Predict Churn", type="primary"
                    )

            with col2:
                st.subheader("🔮 Prediction Results")

                if submitted_class:
                    with st.spinner(
                        f"🤖 Running predictions with {selected_class_model}..."
                    ):
                        # Prepare input data
                        input_data = {
                            "features": {
                                "age": input_values["age"],
                                "income": input_values["income"],
                                "store_location": input_values["store_location"],
                                "customer_city": input_values["customer_city"],
                                "store_city": input_values["store_city"],
                                "total_spent": input_values["total_spent"],
                                "num_purchases": input_values["num_purchases"],
                                "online_purchases": input_values["online_purchases"],
                                "in_store_purchases": input_values[
                                    "in_store_purchases"
                                ],
                                "avg_purchase": input_values["avg_purchase"],
                                "days_since_last_purchase": input_values[
                                    "days_since_last_purchase"
                                ],
                                "website_visits": input_values["website_visits"],
                                "has_complained": input_values["has_complained"],
                                "is_active_member": input_values["is_active_member"],
                                "email_subscriptions": input_values[
                                    "email_subscriptions"
                                ],
                                "distance_to_store": input_values["distance_to_store"],
                                "product_return_rate": input_values[
                                    "product_return_rate"
                                ],
                                "quantity": input_values["quantity"],
                                "unit_price": input_values["unit_price"],
                                "discount_applied": input_values["discount_applied"],
                                "app_usage_frequency": input_values[
                                    "app_usage_frequency"
                                ],
                                "social_media_engagement": input_values[
                                    "social_media_engagement"
                                ],
                            },
                            "model_name": selected_class_model,
                            "model_type": "classification",
                        }

                        try:
                            # Make prediction request
                            response = requests.post(
                                f"{API_URL}/predict", json=input_data
                            )

                            if response.status_code == 200:
                                prediction = response.json()

                                # Display churn prediction
                                st.subheader("🎯 Churn Prediction")
                                prediction_label = prediction.get("label", "Unknown")
                                churn_prob = prediction.get("probability", 0.5)

                                # Display prediction result
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("Prediction", prediction_label)
                                with c2:
                                    st.metric("Probability", f"{churn_prob:.2%}")

                                # Create a gauge chart for churn probability
                                fig_churn = go.Figure(
                                    go.Indicator(
                                        mode="gauge+number+delta",
                                        value=churn_prob * 100,
                                        domain={"x": [0, 1], "y": [0, 1]},
                                        title={"text": "Churn Probability (%)"},
                                        delta={"reference": 50},
                                        gauge={
                                            "axis": {"range": [None, 100]},
                                            "bar": {"color": "darkred"},
                                            "steps": [
                                                {
                                                    "range": [0, 25],
                                                    "color": "lightgreen",
                                                },
                                                {"range": [25, 50], "color": "yellow"},
                                                {"range": [50, 75], "color": "orange"},
                                                {"range": [75, 100], "color": "red"},
                                            ],
                                            "threshold": {
                                                "line": {"color": "red", "width": 4},
                                                "thickness": 0.75,
                                                "value": 50,
                                            },
                                        },
                                    )
                                )
                                fig_churn.update_layout(height=300)
                                st.plotly_chart(fig_churn, width="stretch")

                                # Churn risk assessment
                                if churn_prob < 0.3:
                                    st.success(
                                        "✅ Low Risk - Customer is likely to stay"
                                    )
                                elif churn_prob < 0.6:
                                    st.warning(
                                        "⚠️ Medium Risk - Monitor customer closely"
                                    )
                                else:
                                    st.error(
                                        "🚨 High Risk - Immediate intervention needed"
                                    )

                                # Model Explainability
                                st.subheader("🔍 Feature Importance")

                                # Get feature importance
                                feature_importance_data = fetch_feature_importance(
                                    selected_class_model
                                )

                                if (
                                    feature_importance_data
                                    and "feature_importance" in feature_importance_data
                                    and feature_importance_data["feature_importance"]
                                ):
                                    importance_data = feature_importance_data[
                                        "feature_importance"
                                    ]
                                    importance_df = pd.DataFrame(importance_data)

                                    fig_imp = px.bar(
                                        importance_df,
                                        x="importance",
                                        y="feature",
                                        orientation="h",
                                        title=f"Feature Importance ({selected_class_model})",
                                        color="importance",
                                        color_continuous_scale="Blues",
                                    )
                                    fig_imp.update_layout(height=300)
                                    st.plotly_chart(fig_imp, width="stretch")
                                else:
                                    st.info(
                                        "Feature importance not available for this model."
                                    )

                            else:
                                st.error(f"Error making prediction: {response.text}")
                        except Exception as e:
                            st.error(f"Error connecting to backend: {e}")

        # --- Regression Tab ---
        if prediction_type == "CLV Prediction (Regression)":
            st.header("💰 CLV Prediction")

            # Create two columns for input and results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🤷‍♂️ Customer Input")
                with st.form("regression_form"):
                    # Model Selection
                    reg_models = available_models.get("regression", [])
                    if not reg_models:
                        reg_models = ["linear_regression_regression"]

                    selected_reg_model = st.selectbox(
                        "Select Regression Model",
                        reg_models,
                        index=0,
                        key="reg_model_select",
                    )

                    input_values = render_prediction_form("reg", location_options)
                    submitted_reg = st.form_submit_button(
                        "🚀 Predict Revenue", type="primary"
                    )

            with col2:
                st.subheader("🔮 Prediction Results")

                if submitted_reg:
                    with st.spinner(
                        f"🤖 Running predictions with {selected_reg_model}..."
                    ):
                        # Prepare input data
                        input_data = {
                            "features": {
                                "age": input_values["age"],
                                "income": input_values["income"],
                                "store_location": input_values["store_location"],
                                "customer_city": input_values["customer_city"],
                                "store_city": input_values["store_city"],
                                "total_spent": input_values["total_spent"],
                                "num_purchases": input_values["num_purchases"],
                                "online_purchases": input_values["online_purchases"],
                                "in_store_purchases": input_values[
                                    "in_store_purchases"
                                ],
                                "avg_purchase": input_values["avg_purchase"],
                                "days_since_last_purchase": input_values[
                                    "days_since_last_purchase"
                                ],
                                "website_visits": input_values["website_visits"],
                                "has_complained": input_values["has_complained"],
                                "is_active_member": input_values["is_active_member"],
                                "email_subscriptions": input_values[
                                    "email_subscriptions"
                                ],
                                "distance_to_store": input_values["distance_to_store"],
                                "product_return_rate": input_values[
                                    "product_return_rate"
                                ],
                                "quantity": input_values["quantity"],
                                "unit_price": input_values["unit_price"],
                                "discount_applied": input_values["discount_applied"],
                                "app_usage_frequency": input_values[
                                    "app_usage_frequency"
                                ],
                                "social_media_engagement": input_values[
                                    "social_media_engagement"
                                ],
                            },
                            "model_name": selected_reg_model,
                            "model_type": "regression",
                        }

                        try:
                            # Make prediction request
                            response = requests.post(
                                f"{API_URL}/predict", json=input_data
                            )

                            if response.status_code == 200:
                                prediction = response.json()

                                # Display revenue prediction
                                st.subheader("💰 Revenue Prediction")
                                predicted_revenue = prediction.get("prediction", 0)
                                confidence = prediction.get("confidence", 0.8)

                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric(
                                        "Predicted Annual Revenue",
                                        f"${predicted_revenue:,.2f}",
                                        delta=f"{confidence*100:.1f}% confidence",
                                    )

                                with c2:
                                    # Revenue range based on confidence
                                    lower_bound = predicted_revenue * (
                                        1 - (1 - confidence)
                                    )
                                    upper_bound = predicted_revenue * (
                                        1 + (1 - confidence)
                                    )
                                    st.metric(
                                        "Revenue Range",
                                        f"${lower_bound:,.0f} - ${upper_bound:,.0f}",
                                    )

                                # Model Explainability
                                st.subheader("🔍 Feature Importance")

                                # Get feature importance
                                feature_importance_data = fetch_feature_importance(
                                    selected_reg_model
                                )

                                if (
                                    feature_importance_data
                                    and "feature_importance" in feature_importance_data
                                    and feature_importance_data["feature_importance"]
                                ):
                                    importance_data = feature_importance_data[
                                        "feature_importance"
                                    ]
                                    importance_df = pd.DataFrame(importance_data)

                                    fig_imp = px.bar(
                                        importance_df,
                                        x="importance",
                                        y="feature",
                                        orientation="h",
                                        title=f"Feature Importance ({selected_reg_model})",
                                        color="importance",
                                        color_continuous_scale="Blues",
                                    )
                                    fig_imp.update_layout(height=300)
                                    st.plotly_chart(fig_imp, width="stretch")
                                else:
                                    st.info(
                                        "Feature importance not available for this model."
                                    )

                            else:
                                st.error(f"Error making prediction: {response.text}")
                        except Exception as e:
                            st.error(f"Error connecting to backend: {e}")

elif dashboard_section == "🗺️ Geographic Visualization":
    st.title("🗺️ Geographic Visualization")

    # Fetch geographic data
    geo_data = fetch_geo_data()

    if not geo_data:
        st.warning("⚠️ No geographic data available. Showing demo visualization.")

        # Create demo geographic data
        demo_geo_data = [
            {
                "country": "USA",
                "city": "New York",
                "customers": 1250,
                "churn_rate": 0.23,
                "avg_revenue": 45000,
            },
            {
                "country": "USA",
                "city": "Los Angeles",
                "customers": 980,
                "churn_rate": 0.19,
                "avg_revenue": 42000,
            },
            {
                "country": "USA",
                "city": "Chicago",
                "customers": 750,
                "churn_rate": 0.27,
                "avg_revenue": 38000,
            },
            {
                "country": "Canada",
                "city": "Toronto",
                "customers": 650,
                "churn_rate": 0.15,
                "avg_revenue": 48000,
            },
            {
                "country": "Canada",
                "city": "Vancouver",
                "customers": 420,
                "churn_rate": 0.18,
                "avg_revenue": 52000,
            },
            {
                "country": "UK",
                "city": "London",
                "customers": 890,
                "churn_rate": 0.22,
                "avg_revenue": 46000,
            },
            {
                "country": "UK",
                "city": "Manchester",
                "customers": 340,
                "churn_rate": 0.25,
                "avg_revenue": 35000,
            },
            {
                "country": "Germany",
                "city": "Berlin",
                "customers": 560,
                "churn_rate": 0.20,
                "avg_revenue": 44000,
            },
            {
                "country": "Germany",
                "city": "Munich",
                "customers": 380,
                "churn_rate": 0.16,
                "avg_revenue": 51000,
            },
            {
                "country": "France",
                "city": "Paris",
                "customers": 720,
                "churn_rate": 0.21,
                "avg_revenue": 47000,
            },
        ]

        geo_df = pd.DataFrame(demo_geo_data)
    else:
        geo_df = pd.DataFrame(geo_data)

    # Geographic overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_customers = geo_df["customers"].sum()
        st.metric("👥 Total Customers", f"{total_customers:,}")

    with col2:
        avg_churn_rate = geo_df["churn_rate"].mean()
        st.metric("📊 Avg Churn Rate", f"{avg_churn_rate:.1%}")

    with col3:
        avg_revenue = geo_df["avg_revenue"].mean()
        st.metric("💰 Avg Revenue", f"${avg_revenue:,.0f}")

    with col4:
        total_countries = geo_df["country"].nunique()
        st.metric("🌍 Countries", total_countries)

    # Geographic visualizations
    st.subheader("📊 Geographic Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Customer distribution by country
        country_summary = (
            geo_df.groupby("country")
            .agg({"customers": "sum", "churn_rate": "mean", "avg_revenue": "mean"})
            .reset_index()
        )

        fig_customers = px.bar(
            country_summary,
            x="country",
            y="customers",
            title="👥 Customer Distribution by Country",
            color="customers",
            color_continuous_scale="Blues",
        )
        fig_customers.update_layout(height=400)
        st.plotly_chart(fig_customers, width="stretch")

    with col2:
        # Churn rate by country
        fig_churn = px.bar(
            country_summary,
            x="country",
            y="churn_rate",
            title="📈 Churn Rate by Country",
            color="churn_rate",
            color_continuous_scale="Reds",
            text="churn_rate",
        )
        fig_churn.update_traces(texttemplate="%{text:.1%}")
        fig_churn.update_layout(height=400)
        st.plotly_chart(fig_churn, width="stretch")

    # City-level analysis
    st.subheader("🏙️ City-Level Analysis")

    # Top cities by customers
    top_cities = geo_df.nlargest(10, "customers")

    fig_top_cities = px.scatter(
        top_cities,
        x="customers",
        y="churn_rate",
        size="avg_revenue",
        color="country",
        hover_data=["city"],
        title="🏙️ Top Cities: Customers vs Churn Rate (Bubble size = Avg Revenue)",
        height=500,
    )
    st.plotly_chart(fig_top_cities, width="stretch")

    # Detailed city table
    st.subheader("📋 City Details")

    # Format the dataframe
    display_geo_df = geo_df.copy()
    display_geo_df["churn_rate"] = display_geo_df["churn_rate"].apply(
        lambda x: f"{x:.1%}"
    )
    display_geo_df["avg_revenue"] = display_geo_df["avg_revenue"].apply(
        lambda x: f"${x:,.0f}"
    )
    display_geo_df["customers"] = display_geo_df["customers"].apply(lambda x: f"{x:,}")

    st.dataframe(display_geo_df, width="stretch")

    # Geographic insights
    st.subheader("🔍 Geographic Insights")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        # Best performing countries (lowest churn)
        best_countries = country_summary.nsmallest(3, "churn_rate")
        st.markdown("**🟢 Best Performing Countries (Lowest Churn):**")
        for _, row in best_countries.iterrows():
            st.success(f"🏆 {row['country']}: {row['churn_rate']:.1%} churn rate")

    with insights_col2:
        # Countries needing attention (highest churn)
        attention_countries = country_summary.nlargest(3, "churn_rate")
        st.markdown("**🔴 Countries Needing Attention (Highest Churn):**")
        for _, row in attention_countries.iterrows():
            st.error(f"⚠️ {row['country']}: {row['churn_rate']:.1%} churn rate")

# Business Insights
elif dashboard_section == "📈 Business Insights":
    st.title("📈 Business Insights")

    # Fetch real data from backend
    revenue_at_risk_data = fetch_revenue_at_risk()
    customer_segments_data = fetch_customer_segments()

    # Business metrics calculations
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Revenue at risk calculation
        if revenue_at_risk_data and "total_revenue_at_risk" in revenue_at_risk_data:
            total_revenue_at_risk = revenue_at_risk_data["total_revenue_at_risk"]
            st.metric("💰 Revenue at Risk", f"${total_revenue_at_risk:,.0f}")
        else:
            st.metric("💰 Revenue at Risk", "$56.25M")

    with col2:
        # Customer lifetime value from segments
        if customer_segments_data and "total_revenue" in customer_segments_data:
            total_revenue = customer_segments_data["total_revenue"]
            total_customers = customer_segments_data["total_customers"]
            avg_clv = (total_revenue / total_customers) * 3  # Assume 3 year lifespan
            st.metric("📊 Avg Customer Lifetime Value", f"${avg_clv:,.0f}")
        else:
            st.metric("📊 Avg Customer Lifetime Value", "$125,000")

    with col3:
        # Campaign effectiveness (placeholder - will be calculated when user uses ROI calculator)
        st.metric("📈 Campaign ROI", "2.3x")

    with col4:
        # Customer acquisition cost (placeholder)
        st.metric("💳 Customer Acquisition Cost", "$450")

    # Business insights sections
    st.subheader("📊 Business Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Monthly revenue trends (mock data)
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        revenue_trend = [
            42000000,
            43500000,
            44800000,
            46200000,
            47500000,
            48900000,
            50200000,
            51600000,
            52900000,
            54300000,
            55600000,
            57000000,
        ]

        revenue_df = pd.DataFrame({"Month": months, "Revenue": revenue_trend})

        fig_revenue = px.line(
            revenue_df,
            x="Month",
            y="Revenue",
            title="📈 Monthly Revenue Trends",
            markers=True,
        )
        fig_revenue.update_layout(height=400)
        st.plotly_chart(fig_revenue, width="stretch")

    with col2:
        # Customer segments performance
        segments = ["High Value", "Medium Value", "Low Value", "At Risk"]
        segment_revenue = [180000, 85000, 25000, 15000]
        segment_customers = [1200, 2800, 3500, 1500]

        segment_df = pd.DataFrame(
            {
                "Segment": segments,
                "Avg Revenue": segment_revenue,
                "Customers": segment_customers,
            }
        )

        fig_segments = px.bar(
            segment_df,
            x="Segment",
            y="Avg Revenue",
            title="💰 Customer Segment Performance",
            color="Avg Revenue",
            color_continuous_scale="Greens",
        )
        fig_segments.update_layout(height=400)
        st.plotly_chart(fig_segments, width="stretch")

    # Campaign ROI Calculator
    st.subheader("🧮 Campaign ROI Calculator")

    with st.expander("📊 Calculate Campaign ROI"):
        col1, col2 = st.columns(2)

        with col1:
            campaign_budget = st.number_input(
                "Campaign Budget ($)", 1000, 1000000, 50000, step=1000
            )
            target_segment = st.selectbox(
                "Target Segment",
                ["high_risk", "medium_risk", "all_customers"],
                format_func=lambda x: x.replace("_", " ").title(),
            )

        with col2:
            if st.button("🚀 Calculate ROI"):
                with st.spinner("Calculating campaign ROI..."):
                    try:
                        # Make API call to calculate ROI
                        response = requests.post(
                            f"{API_URL}/business-insights/campaign-roi",
                            params={
                                "campaign_budget": campaign_budget,
                                "target_segment": target_segment,
                            },
                        )

                        if response.status_code == 200:
                            roi_data = response.json()

                            if "error" not in roi_data:
                                # Display results
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        "📈 Campaign ROI",
                                        f"{roi_data['roi_percentage']:.1f}%",
                                        delta=f"${roi_data['revenue_saved']:,.0f} revenue saved",
                                    )

                                with col2:
                                    st.metric(
                                        "💰 Revenue Saved",
                                        f"${roi_data['revenue_saved']:,.0f}",
                                    )

                                with col3:
                                    st.metric(
                                        "🎯 Expected Retention Improvement",
                                        f"{roi_data['expected_retention_improvement']:.1f}%",
                                    )

                                # Recommendation
                                st.info(
                                    f"**Recommendation:** {roi_data['recommendation']}"
                                )

                                if roi_data.get("payback_period_months"):
                                    st.success(
                                        f"**Payback Period:** {roi_data['payback_period_months']:.1f} months"
                                    )
                            else:
                                st.error(f"❌ Error: {roi_data['error']}")
                        else:
                            st.error(
                                f"❌ Failed to calculate ROI: {response.status_code}"
                            )
                    except Exception as e:
                        st.error(f"❌ Error calculating ROI: {str(e)}")
            else:
                st.info("👆 Click 'Calculate ROI' to see results")

    # Customer Segmentation Analysis
    st.subheader("👥 Customer Segmentation Analysis")

    # Use real customer segments data if available
    if customer_segments_data and "segments" in customer_segments_data:
        segments_df = pd.DataFrame(customer_segments_data["segments"])

        col1, col2 = st.columns(2)

        with col1:
            # Segment distribution pie chart
            fig_segment_dist = px.pie(
                segments_df,
                values="count",
                names="segment",
                title="👥 Customer Segment Distribution",
                color_discrete_map={
                    "Champions": "#2ecc71",
                    "Loyal Customers": "#3498db",
                    "At Risk": "#f39c12",
                    "New Customers": "#9b59b6",
                    "Need Attention": "#e74c3c",
                },
            )
            st.plotly_chart(fig_segment_dist, width="stretch")

        with col2:
            # Revenue by segment
            fig_revenue_segment = px.bar(
                segments_df,
                x="segment",
                y="total_revenue",
                title="💰 Total Revenue by Segment",
                color="total_revenue",
                color_continuous_scale="Greens",
                text="total_revenue",
            )
            fig_revenue_segment.update_traces(texttemplate="$%{text:,.0f}")
            st.plotly_chart(fig_revenue_segment, width="stretch")

        # Segment details table
        st.subheader("📋 Segment Details")

        # Format the dataframe for display
        display_segments_df = segments_df.copy()
        display_segments_df["avg_total_sales"] = display_segments_df[
            "avg_total_sales"
        ].apply(lambda x: f"${x:,.0f}")
        display_segments_df["avg_churn_prob"] = display_segments_df[
            "avg_churn_prob"
        ].apply(lambda x: f"{x:.1%}")
        display_segments_df["total_revenue"] = display_segments_df[
            "total_revenue"
        ].apply(lambda x: f"${x:,.0f}")

        st.dataframe(display_segments_df, width="stretch")

        # Export functionality
        st.subheader("📤 Export Segment Data")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Prepare CSV Export"):
                with st.spinner("Preparing CSV data..."):
                    try:
                        response = requests.get(
                            f"{API_URL}/export/customer-segments",
                            params={"format": "csv"},
                        )
                        if response.status_code == 200:
                            st.session_state["export_csv_data"] = response.content
                            st.success("CSV data ready for download!")
                        else:
                            st.error("Failed to export CSV data")
                    except Exception as e:
                        st.error(f"Error exporting CSV: {str(e)}")

            if "export_csv_data" in st.session_state:
                st.download_button(
                    label="📥 Download CSV",
                    data=st.session_state["export_csv_data"],
                    file_name="customer_segments.csv",
                    mime="text/csv",
                )

        with col2:
            if st.button("📋 Prepare JSON Export"):
                with st.spinner("Preparing JSON data..."):
                    try:
                        response = requests.get(
                            f"{API_URL}/export/customer-segments",
                            params={"format": "json"},
                        )
                        if response.status_code == 200:
                            st.session_state["export_json_data"] = json.dumps(
                                response.json(), indent=2
                            )
                            st.success("JSON data ready for download!")
                        else:
                            st.error("Failed to export JSON data")
                    except Exception as e:
                        st.error(f"Error exporting JSON: {str(e)}")

            if "export_json_data" in st.session_state:
                st.download_button(
                    label="📥 Download JSON",
                    data=st.session_state["export_json_data"],
                    file_name="customer_segments.json",
                    mime="application/json",
                )

    else:
        # Fallback to mock data if real data not available
        st.info(
            "Using demo segmentation data. Connect to backend for real customer segments."
        )

        # Create sample customer segments
        segment_data = []
        for i in range(100):
            segment_data.append(
                {
                    "customer_id": f"CUST_{i+1:04d}",
                    "segment": np.random.choice(
                        ["High Value", "Medium Value", "Low Value", "At Risk"],
                        p=[0.15, 0.35, 0.35, 0.15],
                    ),
                    "age": np.random.randint(25, 70),
                    "income": np.random.randint(30000, 150000),
                    "total_spent": np.random.randint(1000, 100000),
                    "churn_probability": np.random.uniform(0.1, 0.9),
                    "predicted_revenue": np.random.randint(10000, 200000),
                }
            )

        segment_df = pd.DataFrame(segment_data)

        col1, col2 = st.columns(2)

        with col1:
            # Segment distribution pie chart
            segment_counts = segment_df["segment"].value_counts()
            fig_segment_dist = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="👥 Customer Segment Distribution",
            )
            st.plotly_chart(fig_segment_dist, width="stretch")

        with col2:
            # Churn probability by segment
            fig_churn_segment = px.box(
                segment_df,
                x="segment",
                y="churn_probability",
                title="📊 Churn Probability by Segment",
                color="segment",
            )
            st.plotly_chart(fig_churn_segment, width="stretch")

        # Segment summary
        segment_summary = (
            segment_df.groupby("segment")
            .agg(
                {
                    "age": "mean",
                    "income": "mean",
                    "total_spent": "mean",
                    "churn_probability": "mean",
                    "predicted_revenue": "mean",
                }
            )
            .round(2)
        )

        st.subheader("📋 Segment Summary (Demo Data)")
        st.dataframe(segment_summary, width="stretch")

    # Key Business Recommendations
    st.subheader("💡 Key Business Recommendations")

    recommendations = [
        {
            "priority": "🔴 High",
            "recommendation": "Focus retention efforts on 'At Risk' segment (15% of customers)",
            "impact": "Potential to save $12.5M in annual revenue",
        },
        {
            "priority": "🟡 Medium",
            "recommendation": "Upsell campaigns for 'Medium Value' customers",
            "impact": "Could increase average revenue by 25%",
        },
        {
            "priority": "🟢 Low",
            "recommendation": "Referral programs for 'High Value' customers",
            "impact": "Expected to reduce acquisition cost by 30%",
        },
    ]

    for rec in recommendations:
        with st.expander(f"{rec['priority']} Priority: {rec['recommendation']}"):
            st.write(f"**Expected Impact:** {rec['impact']}")
            st.write("**Implementation Timeline:** 3-6 months")
            st.write("**Resource Requirements:** Marketing team + Data Science support")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🚀 Customer Churn MLOps Dashboard | Powered by FastAPI + Streamlit</p>
        <p>📊 Real-time model performance monitoring and business insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)
