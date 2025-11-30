from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import json
import os
from pathlib import Path

app = FastAPI(title="CV Model Service", version="1.0.0")

class PredictionRequest(BaseModel):
    image: str  # base64 encoded image
    ticker: str = "SBER"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

# Global variables for model and label map
model = None
label_map = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_label_map():
    """Load label map from file"""
    label_map_path = Path("../models/CV/label_map.json")
    if label_map_path.exists():
        with open(label_map_path, 'r') as f:
            return json.load(f)
    else:
        # Default mapping
        return {"down": 0, "flat": 1, "up": 2}

def create_model(num_classes=3):
    """Create ResNet50 model for CV classification"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_feat = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes)
    )

    return model

def load_model():
    """Load trained model"""
    global model, label_map

    if model is None:
        try:
            label_map = load_label_map()
            num_classes = len(label_map)

            model = create_model(num_classes)
            model_path = Path("../models/CV/models/best_cv_model.pth")

            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()
                print(f"✓ CV Model loaded successfully. Classes: {label_map}")
            else:
                print(f"⚠ CV Model file not found: {model_path}. Using mock predictions.")
                model = "mock"  # Flag for mock mode
        except Exception as e:
            print(f"⚠ Error loading CV model: {e}. Using mock predictions.")
            model = "mock"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load CV model: {e}")

def preprocess_image(image_base64: str) -> torch.Tensor:
    """Preprocess base64 image for model input"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        # Apply transforms
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(device)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on chart image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Handle mock mode
        if model == "mock":
            import numpy as np
            # Return mock predictions
            classes = ["up", "flat", "down"]
            probs = np.random.dirichlet([2, 2, 2])  # Random but balanced
            predicted_idx = np.argmax(probs)
            prediction = classes[predicted_idx]
            confidence = float(probs[predicted_idx])

            probs_dict = {
                classes[i]: float(probs[i]) for i in range(len(classes))
            }

            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                probabilities=probs_dict
            )

        # Preprocess image
        image_tensor = preprocess_image(request.image)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Convert class index to label
        idx_to_label = {v: k for k, v in label_map.items()}
        prediction = idx_to_label.get(predicted_class, "unknown")

        # Prepare response
        probs_dict = {
            idx_to_label.get(i, f"class_{i}"): float(probabilities[0][i])
            for i in range(len(probabilities[0]))
        }

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probs_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "mock" if model == "mock" else "pytorch",
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
