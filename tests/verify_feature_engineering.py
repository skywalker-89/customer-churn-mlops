import requests
import json
import time

API_URL = "http://localhost:8000/api"


def test_prediction(features, name, model_name, model_type="classification"):
    payload = {
        "features": features,
        "model_name": model_name,
        "model_type": model_type,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Scenario: {name} | Model: {model_name}")
            if model_type == "classification":
                print(f"Prediction: {result.get('label')}")
                print(f"Probability: {result.get('probability')}")
            else:
                print(f"Prediction: {result.get('prediction')}")
            print("-" * 40)
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Verifying Feature Engineering Impact on Predictions\n")

    # 1. Test Engagement Score Impact (Classification)
    print("\n🧪 Testing Engagement Score Impact (Classification)")

    features_low_engagement = {
        "app_usage_frequency": "Low",
        "social_media_engagement": "Low",
        "email_subscriptions": 0,
        "days_since_last_purchase": 300,
    }

    features_high_engagement = {
        "app_usage_frequency": "High",
        "social_media_engagement": "High",
        "email_subscriptions": 1,
        "days_since_last_purchase": 5,
    }

    # Test with logistic regression
    test_prediction(
        features_low_engagement, "Low Engagement", "logisticregression_classification"
    )
    test_prediction(
        features_high_engagement, "High Engagement", "logisticregression_classification"
    )

    # 2. Test Quantity x Price Impact (Regression)
    # Baseline: Small purchase
    features_small_purchase = {"quantity": 2, "unit_price": 50, "num_purchases": 1}

    # Large purchase
    features_large_purchase = {"quantity": 100, "unit_price": 50, "num_purchases": 1}

    # Large purchase with discount
    features_large_purchase_discount = {
        "quantity": 100,
        "unit_price": 50,
        "num_purchases": 1,
        "discount_applied": 0.5,
    }

    print("\n🧪 Testing Quantity x Price Impact (Regression)")
    test_prediction(
        features_small_purchase,
        "Small Purchase (2 x $50)",
        "xgboost_regression",
        model_type="regression",
    )
    test_prediction(
        features_large_purchase,
        "Large Purchase (100 x $50)",
        "xgboost_regression",
        model_type="regression",
    )
    test_prediction(
        features_large_purchase_discount,
        "Large Purchase with 50% Discount",
        "xgboost_regression",
        model_type="regression",
    )

    # 3. Test Online Preference Impact
    # Online shopper
    features_online = {
        "online_purchases": 10,
        "in_store_purchases": 0,
        "num_purchases": 10,
        "total_spent": 1000,
    }

    # In-store shopper
    features_instore = {
        "online_purchases": 0,
        "in_store_purchases": 10,
        "num_purchases": 10,
        "total_spent": 1000,
    }

    # print("\n🧪 Testing Online Preference Impact (Classification)")
    # test_prediction(features_online, "Online Shopper", "randomforest_classification")
    # test_prediction(features_instore, "In-Store Shopper", "randomforest_classification")
