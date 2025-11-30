from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

app = FastAPI(title="ML Model Service", version="1.0.0")

class PredictionRequest(BaseModel):
    data: dict  # Dictionary containing time, open, high, low, close, volume data

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    features_used: list

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
scaler = None
feature_columns = None

class FinancialSSM(nn.Module):
    """SSM Model architecture matching the training script"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config['input_dim'], config['d_model'])

        self.gru = nn.GRU(
            input_size=config['d_model'],
            hidden_size=config['d_model'],
            num_layers=config['n_layers'],
            batch_first=True,
            dropout=config['dropout_rate'] if config['n_layers'] > 1 else 0
        )

        self.norm = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.classifier = nn.Linear(config['d_model'], config['n_classes'])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x, _ = self.gru(x)
        x = self.norm(x[:, -1, :])
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

def load_model():
    """Load the trained ML model"""
    global model, scaler, feature_columns

    try:
        # Load PyTorch SSM model (primary - as requested)
            ssm_path = Path("/models/ML/models/best_ssm_model.pth")
            if ssm_path.exists():
                config = {
                    'input_dim': 12,
                    'd_model': 256,
                    'd_state': 16,
                    'n_layers': 4,
                    'n_classes': 3,
                    'dropout_rate': 0.1
                }
                model = FinancialSSM(config)
                checkpoint = torch.load(ssm_path, map_location=device)
                model.load_state_dict(checkpoint)
                model.to(device)
                model.eval()
                print("✓ Loaded PyTorch SSM model")
            else:
                raise FileNotFoundError("No ML model found")

        # Load scaler if it exists
            scaler_path = Path("/models/ML/models/scaler.pkl")
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                print("✓ Loaded feature scaler")
            else:
                print("⚠ No scaler found, using raw features")

        # Define feature columns (matching trained model)
            feature_columns = [
                'close', 'volume', 'sma_ratio', 'rsi', 'boll_pos', 'boll_std',
                'momentum_5', 'log_ret', 'atr', 'label-1', 'label-2', 'label-3'
            ]

    except Exception as e:
        print(f"Error loading ML model: {e}")
        model = None

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for ML model"""
    if len(df) < 20:  # Need minimum data for indicators
        raise ValueError("Insufficient data for feature computation")

    df = df.copy().reset_index(drop=True)

    # Basic price features
    df['log_ret'] = np.log(df['close']).diff()
    df['ret_1'] = df['log_ret'].shift(1)

    # Moving averages
    for w in [3, 5, 10]:
        df[f'sma_{w}'] = df['close'].rolling(window=w).mean()
    df['sma_ratio'] = df['sma_3'] / df['sma_10']

    # RSI (simplified)
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi'] = compute_rsi(df['close'])

    # Bollinger Bands
    mid = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['boll_up'] = mid + 2 * std
    df['boll_low'] = mid - 2 * std
    df['boll_pos'] = (df['close'] - df['boll_low']) / (df['boll_up'] - df['boll_low'])
    df['boll_std'] = std  # Add boll_std feature

    # Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1

    # ATR (simplified)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # Lag features for labels (if available)
    if 'label' in df.columns:
        for i in range(1, 4):
            df[f'label-{i}'] = df['label'].shift(i)

    return df

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using ML model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        # Compute features
        df_features = compute_features(df)

        if len(df_features) < 20:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")

        # Get latest features
        latest_row = df_features.iloc[-1:].copy()

        # Handle missing label features (set to 0 for inference)
        for label_col in ['label-1', 'label-2', 'label-3']:
            if label_col not in latest_row.columns:
                latest_row[label_col] = 0.0

        latest_features = latest_row[feature_columns]

        if scaler:
            features_scaled = scaler.transform(latest_features.values)
        else:
            features_scaled = latest_features.values

        # Make prediction (PyTorch SSM model)
        # Model expects sequences of length 60
        sequence_length = 60

        with torch.no_grad():
            # Create sequence by repeating current features 60 times
            sequence = np.tile(features_scaled, (sequence_length, 1))  # (60, 12)
            inputs = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # (1, 60, 12)
            print(f"Input shape: {inputs.shape}")
            outputs = model(inputs)
            print(f"Model output shape: {outputs.shape}")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            print(f"Probabilities shape: {probabilities.shape}, values: {probabilities}")
            predicted_class = np.argmax(probabilities)
            print(f"Predicted class: {predicted_class}")

        confidence = float(probabilities[predicted_class])

        # Map to labels (handle both 2-class and 3-class models)
        if len(probabilities) == 2:
            # Binary classification: up/down
            label_map = {0: 'down', 1: 'up'}
            prediction = label_map.get(predicted_class, 'unknown')
            probs_dict = {
                'up': float(probabilities[1]),
                'flat': 0.0,  # No flat prediction for binary
                'down': float(probabilities[0])
            }
        else:
            # 3-class classification
            label_map = {0: 'up', 1: 'flat', 2: 'down'}
            prediction = label_map.get(predicted_class, 'unknown')
            probs_dict = {
                'up': float(probabilities[0]),
                'flat': float(probabilities[1]),
                'down': float(probabilities[2])
            }

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probs_dict,
            features_used=feature_columns
        )

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(f"Available columns in df_features: {list(df_features.columns)}")
        print(f"Expected features: {feature_columns}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
