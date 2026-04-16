"""
UI Components Module
Reusable Streamlit UI components
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

from .config import CONSTRAINTS, CONTRACT_LENGTH_OPTIONS, APP_MODES
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
    def render_sidebar(feature_importance: Optional[dict] = None) -> str:
        """
        Render sidebar navigation and optional feature importance chart.

        Returns:
            Selected app mode
        """
        st.sidebar.header("Navigation")
        app_mode = st.sidebar.selectbox("Choose Mode", APP_MODES)

        # Feature Importance Section
        st.sidebar.markdown("---")
        st.sidebar.header("Model Feature Importance")
        if feature_importance:
            # Collapse one-hot encoded contract length back to a single 'Contract Length' score
            contract_score = max(
                feature_importance.get('Contract Length_Annual', 0),
                feature_importance.get('Contract Length_Monthly', 0),
                feature_importance.get('Contract Length_Quarterly', 0)
            )
            display_importance = {
                'Support Calls': feature_importance.get('Support Calls', 0),
                'Payment Delay': feature_importance.get('Payment Delay', 0),
                'Total Spend': feature_importance.get('Total Spend', 0),
                'Contract Length': contract_score,
            }
            total = sum(display_importance.values()) or 1
            display_importance = {k: round(v / total * 100, 1) for k, v in display_importance.items()}

            fi_df = pd.DataFrame(
                list(display_importance.items()),
                columns=['Feature', 'Importance (%)']
            ).sort_values('Importance (%)', ascending=False)

            st.sidebar.dataframe(fi_df.set_index('Feature'), use_container_width=True)
            st.sidebar.bar_chart(fi_df.set_index('Feature'))
            st.sidebar.caption(
                "Importance shows each feature's relative influence on the model's prediction."
            )
        else:
            st.sidebar.info("Feature importance unavailable for this model type.")

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

        # Churn Reasons
        reasons = result.get('churn_reasons', [])
        if reasons:
            st.write("### Potential Churn Reasons")
            if result['is_high_risk']:
                st.write(
                    "Here are the key risk factors contributing to this customer's churn risk:"
                )
            else:
                st.write(
                    "While this customer is low-risk, monitor these factors:"
                )

            severity_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for item in reasons:
                icon = severity_colors.get(item['severity'], '⚪')
                with st.container(border=True):
                    st.markdown(f"**{icon} {item['reason']}**")
                    st.caption(item['detail'])
        elif result['is_high_risk']:
            st.info("No specific dominant risk factors detected — ensemble of inputs led to this prediction.")
    
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
    def render_retention_report(report):
        tier = report.get("risk_tier", "medium")
        prob = report.get("churn_probability", 0.0)
        tier_map = {
            "high": ("🔴 High Risk", "error"),
            "medium": ("🟡 Medium Risk", "warning"),
            "low": ("🟢 Low Risk", "success"),
        }
        label, banner = tier_map.get(tier, ("Medium Risk", "warning"))

        st.subheader("Risk Summary")
        getattr(st, banner)(f"{label} — Churn probability: {prob*100:.1f}%")
        st.write(report.get("risk_summary", ""))

        if report.get("used_fallback"):
            st.info("Language model was unavailable, so a rule-based fallback report was generated.")

        factors = report.get("factors") or []
        if factors:
            st.subheader("Key Factors")
            icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            for f in factors:
                icon = icons.get(f.get("severity"), "⚪")
                with st.container(border=True):
                    st.markdown(f"**{icon} {f.get('name', '')}**")
                    st.caption(f.get("evidence", ""))

        recs = report.get("recommendations") or []
        if recs:
            st.subheader("Recommended Retention Actions")
            for i, r in enumerate(recs, 1):
                with st.expander(f"{i}. {r.get('action', '')}", expanded=(i == 1)):
                    st.markdown(f"**Why:** {r.get('rationale', '')}")
                    st.markdown(f"**Expected impact:** {r.get('expected_impact', '')}")
                    st.markdown(f"**Timeframe:** {r.get('timeframe', '')}")

        sources = report.get("sources") or []
        if sources:
            st.subheader("Supporting Sources")
            for s in sources:
                st.markdown(f"- `{s['source']}` — {s['section']}")

        note = report.get("confidence_note")
        if note:
            st.caption(f"Confidence note: {note}")

        disclaimer = report.get("disclaimer")
        if disclaimer:
            st.markdown("---")
            st.caption(disclaimer)

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
