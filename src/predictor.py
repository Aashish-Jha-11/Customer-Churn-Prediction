"""
Predictor Module
Handles prediction logic and result formatting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

from .model_handler import ModelHandler
from .data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Handles churn predictions"""
    
    def __init__(self, model_handler: ModelHandler):
        """
        Initialize predictor with model handler
        
        Args:
            model_handler: ModelHandler instance
        """
        self.model_handler = model_handler
        self.data_processor = DataProcessor()
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data
        
        Args:
            data: Raw input dataframe
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            # Get model
            model = self.model_handler.get_model()
            if model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess data
            processed_data = self.data_processor.prepare_for_prediction(data)
            
            # Make predictions
            predictions = model.predict(processed_data)
            probabilities = model.predict_proba(processed_data)
            
            logger.info(f"Generated predictions for {len(data)} records")
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_single(self, 
                      total_spend: float,
                      support_calls: int,
                      payment_delay: int,
                      contract_length: str) -> Dict[str, Any]:
        """
        Make prediction for a single customer
        
        Args:
            total_spend: Total amount spent
            support_calls: Number of support calls
            payment_delay: Payment delay in days
            contract_length: Contract type (Monthly/Quarterly/Annual)
            
        Returns:
            Dictionary with prediction results
        """
        # Create input dataframe
        input_data = pd.DataFrame({
            'Total Spend': [total_spend],
            'Support Calls': [support_calls],
            'Payment Delay': [payment_delay],
            'Contract Length': [contract_length]
        })
        
        # Make prediction
        predictions, probabilities = self.predict(input_data)
        
        # Format results
        result = {
            'prediction': int(predictions[0]),
            'prediction_label': 'Will Churn' if predictions[0] == 1 else 'Will Not Churn',
            'churn_probability': float(probabilities[0][1]),
            'retention_probability': float(probabilities[0][0]),
            'is_high_risk': probabilities[0][1] > 0.5
        }
        
        return result
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple customers and return formatted results
        
        Args:
            data: Input dataframe with customer data
            
        Returns:
            DataFrame with original data plus prediction results
        """
        # Make predictions
        predictions, probabilities = self.predict(data)
        
        # Create results dataframe
        results_df = data.copy()
        results_df['Churn Prediction'] = predictions
        results_df['Churn Prediction'] = results_df['Churn Prediction'].map({0: 'No', 1: 'Yes'})
        results_df['Churn Probability (%)'] = (probabilities[:, 1] * 100).round(2)
        results_df['Retention Probability (%)'] = (probabilities[:, 0] * 100).round(2)
        
        return results_df
    
    @staticmethod
    def get_prediction_summary(predictions: np.ndarray, 
                               probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Generate summary statistics for batch predictions
        
        Args:
            predictions: Array of predictions
            probabilities: Array of prediction probabilities
            
        Returns:
            Dictionary with summary statistics
        """
        churn_count = int(np.sum(predictions))
        total_count = len(predictions)
        no_churn_count = total_count - churn_count
        avg_churn_prob = float(probabilities[:, 1].mean())
        
        return {
            'total_customers': total_count,
            'churn_count': churn_count,
            'no_churn_count': no_churn_count,
            'churn_rate': churn_count / total_count if total_count > 0 else 0,
            'avg_churn_probability': avg_churn_prob
        }
