"""
Model Handler Module
Handles loading and caching of the ML model from Hugging Face
"""

import streamlit as st
import joblib
from huggingface_hub import hf_hub_download
from typing import Optional
import logging

from .config import MODEL_REPO_ID, MODEL_FILENAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles model loading and management"""
    
    def __init__(self):
        self.model = None
        
    @staticmethod
    @st.cache_resource
    def load_model():
        """
        Load model from HuggingFace Hub with caching
        
        Returns:
            Loaded model object or None if loading fails
        """
        try:
            logger.info(f"Downloading model from {MODEL_REPO_ID}/{MODEL_FILENAME}")
            
            # Download model from HuggingFace
            model_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME
            )
            
            logger.info(f"Model downloaded to: {model_path}")
            
            # Load the model
            model = joblib.load(model_path)
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def get_model(self):
        """
        Get the loaded model instance
        
        Returns:
            Model object
        """
        if self.model is None:
            self.model = self.load_model()
        return self.model
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is successfully loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.get_model() is not None

