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

st.set_page_config(page_title="EPL Match Outcome Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        letter-spacing: 0.5px;
        line-height: 1.6;
    }
    
    .main {
        background-color: #0a0a0a;
    }
    
    .stApp {
        background-color: #0a0a0a;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
        font-weight: 300;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        letter-spacing: 1px;
        line-height: 1.5;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0a0a0a;
    }
    
    .stButton>button {
        border-radius: 6px;
        border: none;
        background-color: #1a1a1a;
        color: white;
        padding: 0.6rem 1.5rem;
        font-weight: 300;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        letter-spacing: 1px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2d2d2d;
    }
    
    .stSelectbox label, .stDateInput label {
        font-weight: 300;
        color: #e0e0e0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        letter-spacing: 0.5px;
    }
    
    .stMetric {
        background-color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #4a4a4a;
    }
    
    .stMetric label {
        font-weight: 300;
        color: #b0b0b0;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-weight: 300;
        color: #e0e0e0;
        letter-spacing: 0.5px;
    }
    
    .sidebar .sidebar-content {
        background-color: #0a0a0a;
    }
    
    .sidebar h1, .sidebar h2, .sidebar h3 {
        color: #ffffff;
    }
    
    .sidebar label {
        color: #e0e0e0;
    }
    
    .stSuccess {
        background-color: #1a1a1a;
        color: white;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #1a1a1a;
        color: white;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #1a1a1a;
        color: white;
        border-radius: 6px;
        padding: 1rem;
    }
    
    hr {
        border: none;
        height: 1px;
        background-color: #2d2d2d;
        margin: 2rem 0;
    }
    
    body {
        color: #e0e0e0;
        background-color: #0a0a0a;
        letter-spacing: 0.5px;
        line-height: 1.6;
    }
    
    p, div, span {
        color: #e0e0e0;
        letter-spacing: 0.5px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.title("English Premier League Match Outcome Predictor")

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
    
    predict_button = st.button("Predict Outcome", type="primary")

if predict_button:
    try:
        with st.spinner("Making prediction..."):
            result = predict_match(home_team, away_team, datetime.combine(match_date, datetime.min.time()))
        
        st.success("Prediction Complete")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Outcome", result["predicted_outcome"])
        
        with col2:
            st.metric("Home Win Probability", result["probabilities_percent"]["Home Win"])
        
        with col3:
            st.metric("Draw Probability", result["probabilities_percent"]["Draw"])
        
        st.markdown("<br>", unsafe_allow_html=True)
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
            color_continuous_scale="Viridis",
            text="Probability"
        )
        fig.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            marker_line_color="rgba(0,0,0,0.2)",
            marker_line_width=1.5
        )
        fig.update_layout(
            yaxis_title="Probability",
            yaxis_tickformat=".1%",
            showlegend=False,
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#e0e0e0"),
            xaxis=dict(title_font=dict(family="Inter, sans-serif", size=13), tickfont=dict(color="#e0e0e0")),
            yaxis=dict(title_font=dict(family="Inter, sans-serif", size=13), tickfont=dict(color="#e0e0e0"))
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError as e:
        if "xgboost_epl_match_outcome.pkl" in str(e):
            st.error("Model file not found")
            st.markdown("""
            <div style='background-color: #1a1a1a; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;'>
            <p style='color: #e0e0e0; margin-bottom: 1rem;'><strong>To create the model:</strong></p>
            <ol style='color: #b0b0b0; line-height: 1.8;'>
            <li>Install OpenMP: <code style='background-color: #2d2d2d; padding: 0.2rem 0.5rem; border-radius: 4px;'>brew install libomp</code></li>
            <li>Run: <code style='background-color: #2d2d2d; padding: 0.2rem 0.5rem; border-radius: 4px;'>python create_model.py</code></li>
            </ol>
            <p style='color: #909090; font-size: 0.9rem; margin-top: 1rem;'>
            Note: If you don't have Homebrew, install it from <a href='https://brew.sh' style='color: #4a9eff;'>brew.sh</a>
            </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Ensure the model has been trained and historical data is available.")

st.divider()

st.subheader("Feature Importances")
st.markdown("""
<div style='color: #b0b0b0; margin-bottom: 1.5rem; letter-spacing: 0.5px; line-height: 1.8;'>
Analysis of model feature weights and their contribution to predictions
</div>
""", unsafe_allow_html=True)

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
        color_continuous_scale="Viridis"
    )
    fig.update_traces(
        marker_line_color="rgba(0,0,0,0.2)",
        marker_line_width=1.5
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=11, color="#e0e0e0"),
        xaxis=dict(title_font=dict(family="Inter, sans-serif", size=12), tickfont=dict(color="#e0e0e0")),
        yaxis=dict(title_font=dict(family="Inter, sans-serif", size=12), tickfont=dict(color="#e0e0e0"))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View all features"):
        st.dataframe(feature_importance_df, use_container_width=True)
        
except FileNotFoundError:
    st.info("Feature importances not available. Run the model training pipeline first.")

