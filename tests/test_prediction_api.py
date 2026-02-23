import requests
import json

API_URL = "http://localhost:8000/api"


def test_prediction(features, name, model_name="logisticregression_classification"):
    payload = {
        "features": features,
        "model_name": model_name,
        "model_type": "classification",
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Scenario: {name} | Model: {model_name}")
            print(f"Prediction: {result.get('label')}")
            print(f"Probability: {result.get('probability')}")
            print("-" * 40)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    # Scenario 1: Loyal Customer
    features_loyal = {
        "age": 45,
        "income": 80000,
        "num_purchases": 50,
        "total_spent": 5000,
        "days_since_last_purchase": 5,
        "has_complained": 0,
        "is_active_member": 1,
        "avg_purchase": 100,  # Added for mapping test
    }

    # Scenario 2: Churn Risk
    features_churn = {
        "age": 25,
        "income": 20000,
        "num_purchases": 1,
        "total_spent": 50,
        "days_since_last_purchase": 100,
        "has_complained": 1,
        "is_active_member": 0,
        "avg_purchase": 50,  # Added for mapping test
    }

    # Scenario 3: Very High Churn Risk
    features_very_high_churn = {
        "age": 30,
        "income": 15000,
        "num_purchases": 1,
        "total_spent": 10,
        "days_since_last_purchase": 365,  # 1 year!
        "has_complained": 1,
        "is_active_member": 0,
        "avg_purchase": 10,
    }

    # Test Logistic Regression (Has feature names)
    test_prediction(
        features_loyal, "Loyal Customer", "logisticregression_classification"
    )
    test_prediction(features_churn, "Churn Risk", "logisticregression_classification")
    test_prediction(
        features_very_high_churn,
        "Very High Churn Risk",
        "logisticregression_classification",
    )

    # Scenario 4: High Return Rate (Should be Churn)
    features_return_risk = {
        "age": 30,
        "product_return_rate": 0.8,  # High return rate
        "has_returned_items": 1,
        "total_spent": 500,
        "num_purchases": 5,
    }

    test_prediction(
        features_return_risk, "High Return Rate", "logisticregression_classification"
    )

    # Scenario 5: Regression Test - High Spending
    features_high_spend = {
        "total_spent": 50000,
        "num_purchases": 50,
        "quantity": 500,  # Explicit quantity
        "unit_price": 100,  # Explicit price
    }

    # Scenario 6: Regression Test - Low Spending
    features_low_spend = {
        "total_spent": 100,
        "num_purchases": 1,
        # quantity/unit_price missing, should infer from total_spent
    }

    def test_regression(features, name, model_name="xgboost_regression"):
        payload = {
            "features": features,
            "model_name": model_name,
            "model_type": "regression",
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"Scenario: {name} | Model: {model_name}")
                print(f"Prediction: {result.get('prediction')}")
                print("-" * 40)
            else:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"Connection error: {e}")

    test_regression(features_high_spend, "High Spending", "xgboost_regression")
    test_regression(features_low_spend, "Low Spending", "xgboost_regression")

    print("\nTesting Linear Regression (Scratch):")
    test_regression(
        features_high_spend, "High Spending", "linear_regression_regression"
    )
    test_regression(features_low_spend, "Low Spending", "linear_regression_regression")

    print("\nTesting Random Forest (Sklearn):")
    test_regression(features_high_spend, "High Spending", "random_forest_regression")
    test_regression(features_low_spend, "Low Spending", "random_forest_regression")
