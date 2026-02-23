# Customer Churn Prediction App

A web application for predicting customer churn using machine learning. Built with Streamlit and powered by a Random Forest model hosted on Hugging Face.

## Features

- Single Prediction Mode: Enter individual customer details to get instant churn predictions
- Batch Prediction Mode: Upload CSV files to predict churn for multiple customers at once
- Interactive UI with clear visualizations
- Export prediction results as CSV files
- Modular architecture with clean code structure

## Installation

1. Navigate to the project directory:

   ```bash
   cd Customer-Churn-Prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### Single Prediction Mode

1. Select "Single Prediction" from the sidebar
2. Enter customer details (Total Spend, Support Calls, Payment Delay, Contract Length)
3. Click "Predict Churn" to see results
4. View prediction, probability scores, and visual breakdown

### Batch Prediction Mode

1. Select "Batch Prediction" from the sidebar
2. Upload a CSV file with the required columns:
   - `Total Spend`
   - `Support Calls`
   - `Payment Delay`
   - `Contract Length` (values: Monthly, Quarterly, or Annual)
3. Click "Generate Predictions"
4. Download the results as CSV

Sample CSV format:

```csv
Total Spend,Support Calls,Payment Delay,Contract Length
500.0,2,0,Monthly
1200.0,5,15,Annual
800.0,1,5,Quarterly
```

## Model Information

- Model Type: Random Forest Classifier
- Hosted on: [Hugging Face](https://huggingface.co/manthansubhash01/churn-prediction-model)
- Features: Total Spend, Support Calls, Payment Delay, Contract Length

## Project Structure

```
Customer-Churn-Prediction/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
└── src/                  # Source modules
    ├── config.py         # Configuration and constants
    ├── model_handler.py  # Model loading and management
    ├── data_processor.py # Data preprocessing
    ├── predictor.py      # Prediction logic
    └── ui_components.py  # UI components
```

## Troubleshooting

**Model loading issues:**

- Check your internet connection
- Verify access to https://huggingface.co

**CSV upload errors:**

- Ensure all required columns are present
- Check column names (case-sensitive)
- Verify Contract Length values are: Monthly, Quarterly, or Annual

## Dependencies

- streamlit: Web app framework
- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning utilities
- huggingface-hub: Model loading from Hugging Face
- joblib: Model serialization
