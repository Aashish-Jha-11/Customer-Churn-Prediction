"""
Configuration settings for the Churn Prediction App
"""

# Model Configuration
MODEL_REPO_ID = "manthansubhash01/churn-prediction-model"
MODEL_FILENAME = "churn_model.pkl"

# Feature Configuration 
# Order must match exactly how the model was trained
EXPECTED_FEATURES = [
    'Support Calls',
    'Payment Delay',
    'Total Spend',
    'Contract Length_Annual',
    'Contract Length_Monthly',
    'Contract Length_Quarterly'
]

# Input Features (before encoding) - keep for form display
INPUT_FEATURES = [
    'Total Spend',
    'Support Calls',
    'Payment Delay',
    'Contract Length'
]

# Contract Length Options
CONTRACT_LENGTH_OPTIONS = ["Monthly", "Quarterly", "Annual"]

# UI Configuration
PAGE_TITLE = "Customer Churn Prediction"
PAGE_ICON = "chart_with_upwards_trend"
APP_MODES = ["Single Prediction", "Batch Prediction"]

# Input Constraints
CONSTRAINTS = {
    'total_spend': {
        'min': 0.0,
        'max': 10000.0,
        'default': 500.0,
        'step': 10.0,
        'help': 'Total amount spent by the customer'
    },
    'support_calls': {
        'min': 0,
        'max': 20,
        'default': 2,
        'help': 'Number of support calls made'
    },
    'payment_delay': {
        'min': 0,
        'max': 90,
        'default': 0,
        'help': 'Average payment delay in days'
    }
}

# Prediction Thresholds
CHURN_THRESHOLD = 0.5  # 50% probability
