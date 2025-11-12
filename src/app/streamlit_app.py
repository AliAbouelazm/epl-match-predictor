"""Streamlit app for EPL match outcome prediction."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.prediction_utils import predict_match, load_trained_model
from src.config import CLEANED_DATA_FILE, FEATURE_IMPORTANCE_CSV

st.set_page_config(page_title="EPL Match Outcome Predictor", layout="wide")

st.title("⚽ English Premier League Match Outcome Predictor")
st.markdown("Predict match outcomes using XGBoost machine learning model")

try:
    historical_data = pd.read_csv(CLEANED_DATA_FILE)
    teams = sorted(set(historical_data["home_team"].unique()) | set(historical_data["away_team"].unique()))
except FileNotFoundError:
    st.error("Historical data not found. Please run the data pipeline first.")
    st.stop()

with st.sidebar:
    st.header("Match Details")
    home_team = st.selectbox("Home Team", teams)
    away_team = st.selectbox("Away Team", [t for t in teams if t != home_team])
    match_date = st.date_input("Match Date", value=date.today())
    
    predict_button = st.button("🔮 Predict Outcome", type="primary")

if predict_button:
    try:
        with st.spinner("Making prediction..."):
            result = predict_match(home_team, away_team, datetime.combine(match_date, datetime.min.time()))
        
        st.success("Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Outcome", result["predicted_outcome"])
        
        with col2:
            st.metric("Home Win Probability", result["probabilities_percent"]["Home Win"])
        
        with col3:
            st.metric("Draw Probability", result["probabilities_percent"]["Draw"])
        
        st.subheader("Prediction Probabilities")
        
        prob_df = pd.DataFrame([
            {"Outcome": "Home Win", "Probability": result["probabilities"]["Home Win"]},
            {"Outcome": "Draw", "Probability": result["probabilities"]["Draw"]},
            {"Outcome": "Away Win", "Probability": result["probabilities"]["Away Win"]}
        ])
        
        fig = px.bar(
            prob_df,
            x="Outcome",
            y="Probability",
            color="Probability",
            color_continuous_scale="RdYlGn",
            text="Probability"
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(
            yaxis_title="Probability",
            yaxis_tickformat=".1%",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Ensure the model has been trained and historical data is available.")

st.divider()

st.subheader("📊 Feature Importances")
st.markdown("These are the most important features used by the model to make predictions.")

try:
    feature_importance_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    top_features = feature_importance_df.head(10)
    
    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance",
        color_continuous_scale="viridis"
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View all features"):
        st.dataframe(feature_importance_df, use_container_width=True)
        
except FileNotFoundError:
    st.info("Feature importances not available. Run the model training pipeline first.")

st.divider()

st.markdown("""
### About This Model

This predictor uses an XGBoost classifier trained on historical EPL match data. 
The model considers team form, recent performance, and home/away statistics to predict match outcomes.

**Note:** These predictions are based on historical patterns and should not be used for betting purposes.
""")

