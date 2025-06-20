from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List
import os

app = FastAPI(
    title="🔒 Credit Card Fraud Detection API",
    description="ML-powered fraud detection system for credit card transactions",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
try:
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    print("✅ Models loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("🔄 Please run train_model.py first")
    model = scaler = feature_names = None

class Transaction(BaseModel):
    """Transaction data model"""
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount", gt=0)

class PredictionResponse(BaseModel):
    """Prediction response model"""
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    transaction_id: str = None

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "🔒 Credit Card Fraud Detection API",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "api_status": "healthy",
        "model_status": "loaded" if model else "not_loaded",
        "features_count": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert to DataFrame
        data = transaction.dict()
        df = pd.DataFrame([data])
        
        # Scale Amount feature
        df['Amount'] = scaler.transform(df[['Amount']])
        
        # Ensure correct feature order
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of fraud
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=round(probability, 4),
            risk_level=risk_level,
            transaction_id=f"TXN_{hash(str(data)) % 100000:05d}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "features_count": len(feature_names),
        "feature_names": feature_names[:10]  # First 10 features
    }