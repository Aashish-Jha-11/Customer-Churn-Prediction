import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _tier(prob):
    if prob >= 70:
        return "High"
    if prob >= 40:
        return "Medium"
    return "Low"


def render(results_df):
    df = results_df.copy()
    df["Churn Probability (%)"] = df["Churn Probability (%)"].astype(float)
    df["Risk Tier"] = df["Churn Probability (%)"].apply(_tier)

    total = len(df)
    churners = int((df["Churn Prediction"] == "Yes").sum())
    rate = churners / total if total else 0
    avg_prob = df["Churn Probability (%)"].mean() if total else 0

    st.subheader("Portfolio Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", total)
    c2.metric("Predicted Churners", churners)
    c3.metric("Churn Rate", f"{rate*100:.1f}%")
    c4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Probability Distribution")
        fig = px.histogram(df, x="Churn Probability (%)", nbins=20)
        fig.add_vline(x=50, line_dash="dash", line_color="red")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk Tier Breakdown")
        tier_counts = df["Risk Tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            hole=0.5,
            marker=dict(colors=["#e74c3c", "#f39c12", "#27ae60"]),
        )])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if "Contract Length" in df.columns:
        st.subheader("Churn Rate by Contract Length")
        grouped = df.groupby("Contract Length").apply(
            lambda g: (g["Churn Prediction"] == "Yes").mean() * 100
        ).reset_index(name="Churn Rate (%)")
        fig = px.bar(grouped, x="Contract Length", y="Churn Rate (%)", text="Churn Rate (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if {"Total Spend", "Support Calls", "Payment Delay"}.issubset(df.columns):
        st.subheader("Spend vs Support Calls (risk view)")
        fig = px.scatter(
            df,
            x="Total Spend",
            y="Support Calls",
            color="Churn Probability (%)",
            size="Payment Delay",
            color_continuous_scale="RdYlGn_r",
            hover_data=["Contract Length", "Churn Prediction"],
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if "Total Spend" in df.columns:
        st.subheader("Value vs Risk Segments")
        df["Spend Quartile"] = pd.qcut(
            df["Total Spend"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop"
        )
        pivot = pd.crosstab(df["Spend Quartile"], df["Risk Tier"]).reindex(
            columns=["High", "Medium", "Low"], fill_value=0
        )
        fig = px.imshow(
            pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Risk Tier", y="Spend Quartile", color="Customers"),
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The High-risk x Q4 cell is the High-Value-At-Risk cohort to prioritise.")

    st.subheader("Top 20 At-Risk Customers")
    top = df.sort_values("Churn Probability (%)", ascending=False).head(20)
    st.dataframe(top, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "Download full results as CSV",
        data=csv,
        file_name="churn_analytics.csv",
        mime="text/csv",
    )
