from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any

app = FastAPI(title="RL Model Service", version="1.0.0")

class PredictionRequest(BaseModel):
    data: dict  # Dictionary containing time, open, high, low, close, volume data

class PredictionResponse(BaseModel):
    action: int  # -1 (SELL), 0 (HOLD), 1 (BUY)
    confidence: float
    action_name: str

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

class QNet(nn.Module):
    """Q-Network architecture matching the RL training"""
    def __init__(self, n_obs, n_act=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_act)
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    """Load the trained RL model"""
    global model

    try:
        model_path = Path("../models/rl_SBER_5m_dqn.pt")
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("../models/app/RL/dqn_model.pt"),
                Path("../app/RL/dqn_model.pt"),
                Path("../models/dqn_SBER_5m.pt")
            ]
            for path in alt_paths:
                if path.exists():
                    model_path = path
                    break

        if not model_path.exists():
            print(f"RL model not found at {model_path}. Using mock predictions.")
            model = "mock"  # Flag for mock mode
            return

        # Load the model
        model = QNet(n_obs=6, n_act=3)  # 6 features, 3 actions
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        print(f"✓ RL Model loaded from {model_path}")

    except Exception as e:
        print(f"Error loading RL model: {e}")
        model = None

def compute_rl_features(df_data: dict) -> np.ndarray:
    """Compute features for RL model input"""
    try:
        # Convert to DataFrame for easier processing
        import pandas as pd
        df = pd.DataFrame(df_data)

        if len(df) < 20:
            raise ValueError("Insufficient data for RL features")

        close = df['close'].values
        volume = df['volume'].values

        # Compute technical indicators (matching RL training features)
        # These should match the features used in RL training

        # Simple moving averages
        sma_5 = pd.Series(close).rolling(5).mean().iloc[-1]
        sma_10 = pd.Series(close).rolling(10).mean().iloc[-1]
        sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]

        # RSI (simplified)
        def compute_rsi(prices, period=14):
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]

        rsi = compute_rsi(close)

        # MACD (simplified)
        ema_12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26

        # Bollinger Bands position
        mid = pd.Series(close).rolling(20).mean().iloc[-1]
        std = pd.Series(close).rolling(20).std().iloc[-1]
        boll_pos = (close[-1] - (mid - 2*std)) / (4*std)  # Position within bands

        # Volume ratio
        vol_sma = pd.Series(volume).rolling(20).mean().iloc[-1]
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 1.0

        # Price momentum (5-period)
        momentum = (close[-1] / close[-6]) - 1 if len(close) > 5 else 0.0

        # Create feature vector (adjust based on your RL training features)
        features = np.array([
            close[-1] / sma_5 - 1,  # Price vs SMA5
            rsi / 100,              # Normalized RSI
            macd / close[-1],       # MACD ratio
            boll_pos,               # Bollinger position
            vol_ratio,              # Volume ratio
            momentum                # Momentum
        ], dtype=np.float32)

        return features

    except Exception as e:
        raise ValueError(f"Feature computation failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make RL trading decision"""
    if model is None:
        raise HTTPException(status_code=503, detail="RL model not loaded")

    try:
        # Handle mock mode
        if model == "mock":
            # Return mock predictions based on simple logic
            features = compute_rl_features(request.data)

            # Simple mock logic: if price momentum is positive, suggest BUY
            momentum = features[5]  # momentum is the last feature
            rsi = features[1]       # RSI is the second feature

            if momentum > 0.01 and rsi < 0.7:  # Positive momentum and not overbought
                action = 1  # BUY
                confidence = 0.75
            elif momentum < -0.01 or rsi > 0.8:  # Negative momentum or overbought
                action = -1  # SELL
                confidence = 0.75
            else:
                action = 0  # HOLD
                confidence = 0.60

        else:
            # Compute features
            features = compute_rl_features(request.data)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

            # Get Q-values
            with torch.no_grad():
                q_values = model(features_tensor)[0]

            # Choose action (greedy)
            action_idx = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=0)[action_idx].item()

            # Map action index to action (-1, 0, 1)
            action = action_idx - 1

        # Action names
        action_names = {1: "BUY", -1: "SELL", 0: "HOLD"}
        action_name = action_names.get(action, "HOLD")

        return PredictionResponse(
            action=action,
            confidence=confidence,
            action_name=action_name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "mock" if model == "mock" else "pytorch",
        "device": str(device) if model != "mock" else "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
