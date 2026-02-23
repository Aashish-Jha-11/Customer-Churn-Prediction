"""
UI Components Module
Reusable Streamlit UI components
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from .config import CONSTRAINTS, CONTRACT_LENGTH_OPTIONS
from .data_processor import DataProcessor


class UIComponents:
    """Collection of reusable UI components"""
    
    @staticmethod
    def render_page_header():
        """Render the main page header"""
        st.title("Customer Churn Prediction App")
        st.markdown("""
        This app predicts whether a customer is likely to churn based on their behavior and contract details.
        You can either enter data manually or upload a CSV file for batch predictions.
        """)
    
    @staticmethod
    def render_sidebar() -> str:
        """
        Render sidebar navigation
        
        Returns:
            Selected app mode
        """
        st.sidebar.header("Navigation")
        app_mode = st.sidebar.selectbox("Choose Mode", 
                                         ["Single Prediction", "Batch Prediction"])
        return app_mode
    
    @staticmethod
    def render_single_input_form() -> Dict[str, Any]:
        """
        Render input form for single prediction
        
        Returns:
            Dictionary with input values
        """
        col1, col2 = st.columns(2)
        
        with col1:
            total_spend = st.number_input(
                "Total Spend ($)", 
                min_value=CONSTRAINTS['total_spend']['min'],
                max_value=CONSTRAINTS['total_spend']['max'],
                value=CONSTRAINTS['total_spend']['default'],
                step=CONSTRAINTS['total_spend']['step'],
                help=CONSTRAINTS['total_spend']['help']
            )
            
            support_calls = st.number_input(
                "Support Calls", 
                min_value=CONSTRAINTS['support_calls']['min'],
                max_value=CONSTRAINTS['support_calls']['max'],
                value=CONSTRAINTS['support_calls']['default'],
                help=CONSTRAINTS['support_calls']['help']
            )
            
            payment_delay = st.number_input(
                "Payment Delay (days)", 
                min_value=CONSTRAINTS['payment_delay']['min'],
                max_value=CONSTRAINTS['payment_delay']['max'],
                value=CONSTRAINTS['payment_delay']['default'],
                help=CONSTRAINTS['payment_delay']['help']
            )
            
            contract_length = st.selectbox(
                "Contract Length",
                options=CONTRACT_LENGTH_OPTIONS,
                help="Customer's contract type"
            )
        
        return {
            'total_spend': total_spend,
            'support_calls': support_calls,
            'payment_delay': payment_delay,
            'contract_length': contract_length
        }
    
    @staticmethod
    def render_single_prediction_results(result: Dict[str, Any]):
        """
        Display single prediction results
        
        Args:
            result: Dictionary with prediction results
        """
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        churn_prob_pct = result['churn_probability'] * 100
        retention_prob_pct = result['retention_probability'] * 100
        
        with col1:
            st.metric("Prediction", result['prediction_label'])
        
        with col2:
            st.metric("Churn Probability", f"{churn_prob_pct:.2f}%")
        
        with col3:
            st.metric("Retention Probability", f"{retention_prob_pct:.2f}%")
        
        # Visual indicator
        if result['is_high_risk']:
            st.error("High Risk: This customer is likely to churn. Consider retention strategies!")
        else:
            st.success("Low Risk: This customer is likely to stay.")
        
        # Probability bar chart
        st.write("### Probability Breakdown")
        prob_df = pd.DataFrame({
            'Outcome': ['Will Not Churn', 'Will Churn'],
            'Probability': [retention_prob_pct, churn_prob_pct]
        })
        st.bar_chart(prob_df.set_index('Outcome'))
    
    @staticmethod
    def render_batch_prediction_results(results_df: pd.DataFrame, summary: Dict[str, Any]):
        """
        Display batch prediction results
        
        Args:
            results_df: DataFrame with prediction results
            summary: Summary statistics dictionary
        """
        st.success(f"Predictions completed for {summary['total_customers']} customers!")
        
        # Summary statistics
        st.write("### Prediction Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Customers Likely to Churn", summary['churn_count'])
        
        with col2:
            st.metric("Customers Likely to Stay", summary['no_churn_count'])
        
        with col3:
            avg_prob_pct = summary['avg_churn_probability'] * 100
            st.metric("Average Churn Probability", f"{avg_prob_pct:.2f}%")
        
        # Show results
        st.write("### Detailed Results")
        st.dataframe(results_df)
        
        # Download button
        csv_result = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_result,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Visualization
        st.write("### Churn Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            churn_dist = results_df['Churn Prediction'].value_counts()
            st.bar_chart(churn_dist)
        
        with col2:
            st.write("**Breakdown:**")
            for idx, value in churn_dist.items():
                percentage = (value / len(results_df)) * 100
                st.write(f"- {idx}: {value} customers ({percentage:.1f}%)")
    
    @staticmethod
    def render_csv_format_info():
        """Display expected CSV format information"""
        with st.expander("View Expected CSV Format"):
            st.write("Your CSV file should contain the following columns:")
            sample_df = DataProcessor.create_sample_dataframe()
            st.dataframe(sample_df)
            
            # Download sample CSV
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download Sample CSV",
                data=csv,
                file_name="sample_churn_data.csv",
                mime="text/csv"
            )
    
    @staticmethod
    def render_footer():
        """Render page footer"""
        st.write("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Powered by Hugging Face | Model: manthansubhash01/churn-prediction-model</p>
        </div>
        """, unsafe_allow_html=True)
