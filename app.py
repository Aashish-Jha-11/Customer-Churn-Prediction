"""
Customer Churn Prediction App
Main Streamlit application entry point
"""

import streamlit as st
import pandas as pd

from src.config import PAGE_TITLE, PAGE_ICON
from src.model_handler import ModelHandler
from src.predictor import ChurnPredictor
from src.ui_components import UIComponents

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_app():
    """Initialize application components"""
    model_handler = ModelHandler()
    predictor = ChurnPredictor(model_handler)
    return model_handler, predictor

model_handler, predictor = initialize_app()

# Check if model is loaded
if not model_handler.is_model_loaded():
    st.error("Failed to load model from Hugging Face.")
    st.info("""
    **Troubleshooting:**
    - Check your internet connection
    - Verify the model exists at: https://huggingface.co/manthansubhash01/churn-prediction-model
    - Ensure the file 'churn_model.pkl' is in the repository
    """)
    st.stop()

# Render UI
UIComponents.render_page_header()
app_mode = UIComponents.render_sidebar(
    feature_importance=model_handler.get_feature_importance()
)

# ==================== SINGLE PREDICTION MODE ====================
if app_mode == "Single Prediction":
    st.header("Single Customer Prediction")
    st.write("Enter customer details to predict churn probability.")
    
    # Get input from form
    inputs = UIComponents.render_single_input_form()
    
    st.write("---")
    
    if st.button("Predict Churn", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                # Make prediction
                result = predictor.predict_single(
                    total_spend=inputs['total_spend'],
                    support_calls=inputs['support_calls'],
                    payment_delay=inputs['payment_delay'],
                    contract_length=inputs['contract_length']
                )
                
                # Display results
                UIComponents.render_single_prediction_results(result)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# ==================== BATCH PREDICTION MODE ====================
elif app_mode == "Batch Prediction":
    st.header("Batch Prediction from CSV")
    st.write("Upload a CSV file with customer data to get predictions for multiple customers.")
    
    # Show expected format
    UIComponents.render_csv_format_info()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            
            st.write("### Uploaded Data Preview")
            st.dataframe(data.head(10))
            
            st.write(f"**Total Records:** {len(data)}")
            
            if st.button("Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    try:
                        # Make predictions
                        results_df = predictor.predict_batch(data)
                        
                        # Get summary statistics
                        predictions = (results_df['Churn Prediction'] == 'Yes').astype(int).values
                        probabilities = results_df[['Retention Probability (%)', 
                                                    'Churn Probability (%)']].values / 100
                        summary = ChurnPredictor.get_prediction_summary(predictions, probabilities)
                        
                        # Display results
                        UIComponents.render_batch_prediction_results(results_df, summary)
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
                        st.write("Please make sure your CSV file has the correct format.")
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.write("Please make sure you uploaded a valid CSV file.")

# Footer
UIComponents.render_footer()
