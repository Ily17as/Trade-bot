import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
from datetime import datetime, timedelta
import time

# Configure page
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TINKOFF_API_BASE = "https://api-invest.tinkoff.ru/openapi/v1"
MODEL_SERVICES = {
    'cv': 'http://cv-service:8000',
    'ml': 'http://ml-service:8001',
    'stochastic': 'http://stochastic-service:8002',
    'rl': 'http://rl-service:8003'
}

# Import Tinkoff API
from tinkoff_api import api as tinkoff_api

# Cache data for performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(ticker="SBER", days=30, interval="5min"):
    """Fetch stock data from Tinkoff API"""
    try:
        df = tinkoff_api.fetch_candles(ticker, days=days, interval=interval)
        if df is not None and len(df) > 0:
            return df
        else:
            st.error(f"No data available for {ticker}")
            return None
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def get_model_predictions(model_type, data, ticker="SBER"):
    """Get predictions from model services"""
    try:
        if model_type == 'cv':
            # For CV model, we need to generate a chart image first
            return get_cv_prediction(data, ticker)
        elif model_type == 'ml':
            return get_ml_prediction(data)
        elif model_type == 'stochastic':
            return get_stochastic_prediction(data)
        elif model_type == 'rl':
            return get_rl_prediction(data)
        else:
            return {"error": f"Unknown model type: {model_type}"}
    except Exception as e:
        return {"error": str(e)}

def get_cv_prediction(data, ticker):
    """Get CV model prediction by generating chart image"""
    try:
        # Generate chart image
        import matplotlib.pyplot as plt
        import io
        import base64

        # Create chart image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['time'], data['close'], linewidth=2)
        ax.set_title(f'{ticker} Price Chart')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # Send to CV service
        url = f"{MODEL_SERVICES['cv']}/predict"
        response = requests.post(url, json={'image': image_base64, 'ticker': str(ticker)})

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"CV service returned {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}

def get_ml_prediction(data):
    """Get ML model prediction"""
    try:
        url = f"{MODEL_SERVICES['ml']}/predict"
        # Convert data to JSON-serializable format
        # Convert timestamps to strings
        data_copy = data.copy()
        data_copy['time'] = data_copy['time'].astype(str)
        features = data_copy.to_dict()
        response = requests.post(url, json={'data': features})

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"ML service returned {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}

def get_stochastic_prediction(data):
    """Get stochastic model prediction"""
    try:
        url = f"{MODEL_SERVICES['stochastic']}/predict"
        # Convert data to JSON-serializable format
        data_copy = data.copy()
        data_copy['time'] = data_copy['time'].astype(str)
        response = requests.post(url, json={'data': data_copy.to_dict()})

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Stochastic service returned {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}

def get_rl_prediction(data):
    """Get RL model prediction"""
    try:
        url = f"{MODEL_SERVICES['rl']}/predict"
        # Convert data to JSON-serializable format
        data_copy = data.copy()
        data_copy['time'] = data_copy['time'].astype(str)
        response = requests.post(url, json={'data': data_copy.to_dict()})

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"RL service returned {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}

def create_price_chart(data):
    """Create interactive price chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume'),
        row_width=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data['time'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='OHLC'
    ), row=1, col=1)

    # Volume bar chart
    fig.add_trace(go.Bar(
        x=data['time'],
        y=data['volume'],
        name='Volume',
        marker_color='rgba(0,100,255,0.5)'
    ), row=2, col=1)

    fig.update_layout(
        height=600,
        title_text="Stock Price Chart",
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

def display_model_predictions(data, ticker):
    """Display predictions from all models"""
    st.header("Model Predictions")

    if data is None or len(data) == 0:
        st.warning("No data available for predictions")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Computer Vision Model")
        st.info("Analyzes chart patterns")

        with st.spinner("Getting CV prediction..."):
            cv_result = get_model_predictions('cv', data, ticker)

        if 'error' not in cv_result:
            prediction = cv_result.get('prediction', 'UNKNOWN').upper()
            confidence = cv_result.get('confidence', 0) * 100

            if prediction == 'UP':
                st.success(f"**Prediction: {prediction}**")
                st.metric("Confidence", f"{confidence:.1f}%")
            elif prediction == 'DOWN':
                st.error(f"**Prediction: {prediction}**")
                st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.info(f"**Prediction: {prediction}**")
                st.metric("Confidence", f"{confidence:.1f}%")
        else:
            st.error(f"CV Model Error: {cv_result['error']}")

    with col2:
        st.subheader("Machine Learning Model")
        st.info("Time series analysis")

        with st.spinner("Getting ML prediction..."):
            ml_result = get_model_predictions('ml', data)

        if 'error' not in ml_result:
            prediction = ml_result.get('prediction', 'UNKNOWN').upper()
            confidence = ml_result.get('confidence', 0) * 100

            if prediction == 'UP':
                st.success(f"**Prediction: {prediction}**")
                st.metric("Confidence", f"{confidence:.1f}%")
            elif prediction == 'DOWN':
                st.error(f"**Prediction: {prediction}**")
                st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.info(f"**Prediction: {prediction}**")
                st.metric("Confidence", f"{confidence:.1f}%")
        else:
            st.error(f"ML Model Error: {ml_result['error']}")

    with col3:
        st.subheader("Stochastic Model")
        st.info("Monte Carlo simulation")

        with st.spinner("Getting stochastic prediction..."):
            stoch_result = get_model_predictions('stochastic', data)

        if 'error' not in stoch_result:
            up_prob = stoch_result.get('p_up_tau', 0) * 100
            var95 = stoch_result.get('VaR95', 0) * 100
            score = stoch_result.get('score', 0)

            st.metric("Up Probability", f"{up_prob:.1f}%")
            st.metric("VaR 95%", f"{var95:.2f}%")
            st.metric("Sharpe-like Score", f"{score:.3f}")
        else:
            st.error(f"Stochastic Model Error: {stoch_result['error']}")

    with col4:
        st.subheader("Reinforcement Learning")
        st.info("Trading decisions")

        with st.spinner("Getting RL decision..."):
            rl_result = get_model_predictions('rl', data)

        if 'error' not in rl_result:
            action = rl_result.get('action', 0)
            confidence = rl_result.get('confidence', 0) * 100

            if action == 1:  # BUY
                st.success("**RECOMMENDATION: BUY**")
            elif action == -1:  # SELL
                st.error("**RECOMMENDATION: SELL**")
            else:  # HOLD
                st.info("**RECOMMENDATION: HOLD**")

            st.metric("Action Confidence", f"{confidence:.1f}%")
        else:
            st.error(f"RL Model Error: {rl_result['error']}")

def main():
    st.title("Trading Bot Dashboard")

    # Sidebar controls
    st.sidebar.header("Controls")

    days = st.sidebar.slider("Time Period (days)", 1, 90, 30)

    interval = st.sidebar.selectbox(
        "Time Interval",
        ["1min", "5min", "15min", "1hour", "1day"],
        index=1  # Default to 5min
    )

    # Display market status
    market_status = tinkoff_api.get_market_status()
    if market_status["status"] == "open":
        st.sidebar.success(f"🟢 {market_status['message']}")
    elif market_status["status"] == "closed":
        st.sidebar.warning(f"🔴 {market_status['message']}")
    else:
        st.sidebar.info("⚪ Market status unknown")

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Main content
    with st.spinner("Loading data..."):
        data = get_stock_data("SBER", days, interval)

    if data is not None:
        # Price chart
        st.plotly_chart(create_price_chart(data), use_container_width=True)

        # Model predictions
        display_model_predictions(data, "SBER")

        # Raw data (collapsible)
        with st.expander("View Raw Data"):
            st.dataframe(data.tail(50))

    else:
        st.error("Failed to load stock data. Please check your connection and API keys.")

if __name__ == "__main__":
    main()
