from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import tensorflow as tf

app = FastAPI(
    title="Code Comparison API",
    description="API for comparing original code with Java translations"
)

class CodeComparisonRequest(BaseModel):
    original_code: str
    java_code: str

class CodeComparisonResponse(BaseModel):
    is_equivalent: bool
    confidence: float
    original_code: str
    java_code: str

class BatchComparisonRequest(BaseModel):
    comparisons: List[CodeComparisonRequest]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

model = None
original_code_vectorizer = None
java_code_vectorizer = None

async def initialize_model():
    try:
        global model, original_code_vectorizer, java_code_vectorizer

        model = joblib.load('Transformermodel.pkl')

        original_code_vectorizer = joblib.load('py_vectorizer.pkl')
        java_code_vectorizer = joblib.load('java_vectorizer.pkl')

    except Exception as e:
        raise RuntimeError(f"Failed to initialize model and vectorizers: {str(e)}")

def preprocess_input(original_code: str, java_code: str):

    original_vectorized = original_code_vectorizer(tf.constant([original_code]))
    java_vectorized = java_code_vectorizer(tf.constant([java_code]))
    combined = tf.concat([original_vectorized, java_vectorized], axis=1)
    combined = tf.expand_dims(combined, axis=-1)  
    return combined

    return np.hstack((original_vectorized.toarray(), java_vectorized.toarray()))

@app.post("/predict", response_model=CodeComparisonResponse)
async def predict(request: CodeComparisonRequest):
    try:
        processed_input = preprocess_input(request.original_code, request.java_code)
        prediction = model.predict(processed_input)
        equivalence_probability = float(prediction[0][0]) 
        
        return CodeComparisonResponse(
            is_equivalent=equivalence_probability > 0.5,
            confidence=equivalence_probability,
            original_code=request.original_code,
            java_code=request.java_code
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.on_event("startup")
async def startup_event():
    await initialize_model()