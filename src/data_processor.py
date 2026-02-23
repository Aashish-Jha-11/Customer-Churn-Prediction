"""
Data Processor Module
Handles data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Union, List
import logging

from .config import EXPECTED_FEATURES, INPUT_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles all data preprocessing operations"""
    
    @staticmethod
    def validate_input(data: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate input data has required columns
        
        Args:
            data: Input dataframe
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_cols = [col for col in INPUT_FEATURES if col != 'Contract Length_Annual']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        return True, ""
    
    @staticmethod
    def encode_contract_length(data: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode the Contract Length column
        
        Args:
            data: Input dataframe with 'Contract Length' column
            
        Returns:
            DataFrame with encoded contract length features
        """
        df = data.copy()
        
        if 'Contract Length' in df.columns:
            # One-hot encode Contract Length with space (matching training data)
            df_encoded = pd.get_dummies(df, columns=['Contract Length'], prefix='Contract Length')
        else:
            df_encoded = df.copy()
        
        return df_encoded
    
    @staticmethod
    def add_missing_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add any missing expected features with default values
        
        Args:
            data: Input dataframe
            
        Returns:
            DataFrame with all expected features
        """
        df = data.copy()
        
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
                logger.info(f"Added missing feature: {col}")
        
        return df
    
    @staticmethod
    def select_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the expected features in correct order
        
        Args:
            data: Input dataframe
            
        Returns:
            DataFrame with selected features in correct order
        """
        return data[EXPECTED_FEATURES]
    
    @staticmethod
    def prepare_for_prediction(data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for prediction
        
        Args:
            data: Raw input dataframe
            
        Returns:
            Processed dataframe ready for model prediction
        """
        try:
            # Validate input
            is_valid, error_msg = DataProcessor.validate_input(data)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Encode categorical features
            df_encoded = DataProcessor.encode_contract_length(data)
            
            # Add missing features
            df_complete = DataProcessor.add_missing_features(df_encoded)
            
            # Select features in correct order
            df_final = DataProcessor.select_features(df_complete)
            
            logger.info(f"Successfully processed {len(df_final)} records")
            return df_final
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    @staticmethod
    def create_sample_dataframe() -> pd.DataFrame:
        """
        Create a sample dataframe for demonstration
        
        Returns:
            Sample dataframe
        """
        return pd.DataFrame({
            'Total Spend': [500.0, 1200.0],
            'Support Calls': [2, 5],
            'Payment Delay': [0, 15],
            'Contract Length': ['Monthly', 'Annual']
        })
