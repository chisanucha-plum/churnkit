"""
Test script to demonstrate HIGH RISK predictions.

This script shows examples of customers with high churn probability.
"""

import requests
import json
from typing import Dict, Any


# API endpoint
API_URL = "http://localhost:8000"


def print_prediction(title: str, customer_data: Dict[str, Any]) -> None:
    """
    Make a prediction and print results.
    
    Args:
        title: Title for the prediction
        customer_data: Customer data for prediction
    """
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)
    
    try:
        # Make prediction
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Print results
            print(f"\n✓ Prediction successful!")
            print(f"\n  Churn Probability: {result['churn_probability']:.1%}")
            print(f"  Risk Level:       {result['risk_level']}")
            print(f"  Risk Score:       {result['risk_score']}/100")
            print(f"  Recommendation:   {result['recommendation']}")
            
            # Color coding
            if result['risk_level'] == 'HIGH':
                print(f"\n  🔴 HIGH RISK - Immediate action needed!")
            elif result['risk_level'] == 'MEDIUM':
                print(f"\n  🟡 MEDIUM RISK - Monitor closely")
            else:
                print(f"\n  🟢 LOW RISK - Stable customer")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to API")
        print("  Make sure the server is running: python run.py")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


def main():
    """Run test predictions."""
    
    print("\n" + "=" * 70)
    print("🚀 Customer Churn Prediction - Test Examples")
    print("=" * 70)
    
    # Example 1: HIGH RISK - Month-to-month, new customer, high charges
    print_prediction(
        "Example 1: HIGH RISK - New Month-to-Month Customer",
        {
            "tenure": 2,                          # ⚠️ Very new (2 months)
            "monthly_charges": 110.0,             # ⚠️ High charges
            "total_charges": 220.0,               # Low total (new)
            "contract_type": "Month-to-month",    # ⚠️ Risky contract
            "internet_service": "Fiber optic",    # Expensive service
            "online_security": 0,                 # No security
            "online_backup": 0,                   # No backup
            "device_protection": 0,               # No protection
            "tech_support": 0,                    # No support
            "streaming_tv": 1,
            "streaming_movies": 1,
            "payment_method": "Electronic check",  # ⚠️ Risky payment
            "paperless_billing": 1,
            "senior_citizen": 0,
            "partner": 0,                         # No partner
            "dependents": 0,                      # No dependents
            "phone_service": 1,
            "multiple_lines": 0
        }
    )
    
    # Example 2: HIGH RISK - Electronic check payment, short tenure
    print_prediction(
        "Example 2: HIGH RISK - Electronic Check Payment",
        {
            "tenure": 3,                          # ⚠️ Very new
            "monthly_charges": 95.0,              # ⚠️ High charges
            "total_charges": 285.0,
            "contract_type": "Month-to-month",    # ⚠️ Risky contract
            "internet_service": "Fiber optic",
            "online_security": 0,
            "online_backup": 0,
            "device_protection": 0,
            "tech_support": 0,
            "streaming_tv": 0,
            "streaming_movies": 0,
            "payment_method": "Electronic check",  # ⚠️ Risky payment
            "paperless_billing": 0,
            "senior_citizen": 1,                  # ⚠️ Senior citizen
            "partner": 0,
            "dependents": 0,
            "phone_service": 0,
            "multiple_lines": 0
        }
    )
    
    # Example 3: HIGH RISK - Multiple risk factors
    print_prediction(
        "Example 3: HIGH RISK - Multiple Risk Factors",
        {
            "tenure": 5,                          # ⚠️ Short tenure
            "monthly_charges": 105.0,             # ⚠️ High charges
            "total_charges": 525.0,
            "contract_type": "Month-to-month",    # ⚠️ Risky contract
            "internet_service": "Fiber optic",    # Expensive
            "online_security": 0,                 # No add-ons
            "online_backup": 0,
            "device_protection": 0,
            "tech_support": 0,
            "streaming_tv": 0,
            "streaming_movies": 0,
            "payment_method": "Electronic check",  # ⚠️ Risky payment
            "paperless_billing": 1,
            "senior_citizen": 0,
            "partner": 0,                         # ⚠️ No partner
            "dependents": 0,                      # ⚠️ No dependents
            "phone_service": 0,
            "multiple_lines": 0
        }
    )
    
    # Example 4: MEDIUM RISK - Some protective factors
    print_prediction(
        "Example 4: MEDIUM RISK - Some Protective Factors",
        {
            "tenure": 12,                         # Moderate tenure
            "monthly_charges": 75.0,              # Moderate charges
            "total_charges": 900.0,
            "contract_type": "Month-to-month",    # ⚠️ Risky contract
            "internet_service": "DSL",            # Cheaper service
            "online_security": 1,                 # ✓ Has security
            "online_backup": 1,                   # ✓ Has backup
            "device_protection": 0,
            "tech_support": 1,                    # ✓ Has support
            "streaming_tv": 0,
            "streaming_movies": 0,
            "payment_method": "Bank transfer",    # ✓ Safer payment
            "paperless_billing": 1,
            "senior_citizen": 0,
            "partner": 1,                         # ✓ Has partner
            "dependents": 1,                      # ✓ Has dependents
            "phone_service": 1,
            "multiple_lines": 1
        }
    )
    
    # Example 5: LOW RISK - Long-term customer with contract
    print_prediction(
        "Example 5: LOW RISK - Loyal Customer",
        {
            "tenure": 60,                         # ✓ Long tenure
            "monthly_charges": 65.0,              # ✓ Moderate charges
            "total_charges": 3900.0,              # ✓ High total
            "contract_type": "Two year",          # ✓ Long contract
            "internet_service": "DSL",            # ✓ Stable service
            "online_security": 1,                 # ✓ Has security
            "online_backup": 1,                   # ✓ Has backup
            "device_protection": 1,               # ✓ Has protection
            "tech_support": 1,                    # ✓ Has support
            "streaming_tv": 1,
            "streaming_movies": 1,
            "payment_method": "Credit card",      # ✓ Safe payment
            "paperless_billing": 1,
            "senior_citizen": 0,
            "partner": 1,                         # ✓ Has partner
            "dependents": 1,                      # ✓ Has dependents
            "phone_service": 1,
            "multiple_lines": 1
        }
    )
    
    # Get metrics
    print("\n" + "=" * 70)
    print("📈 Model Metrics")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            print(f"\n✓ Model Performance:")
            print(f"  Accuracy:  {metrics['accuracy']:.1%}")
            print(f"  Precision: {metrics['precision']:.1%}")
            print(f"  Recall:    {metrics['recall']:.1%}")
            print(f"  F1-Score:  {metrics['f1_score']:.1%}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.1%}")
    except Exception as e:
        print(f"✗ Error getting metrics: {str(e)}")
    
    # Get feature importance
    print("\n" + "=" * 70)
    print("🎯 Top 10 Important Features")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/feature-importance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            top_10 = data['top_10']
            
            print("\n✓ Feature Importance:")
            for i, (feature, importance) in enumerate(top_10.items(), 1):
                bar = "█" * int(importance * 50)
                print(f"  {i:2d}. {feature:20s} {bar} {importance:.4f}")
    except Exception as e:
        print(f"✗ Error getting feature importance: {str(e)}")
    
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
