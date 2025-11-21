from fastapi import FastAPI
from pydantic import BaseModel
from src.model_utils import load_model_pipeline, preprocess, decode_predictions, get_triage

app = FastAPI(title="Symptom Checker API")

pipe = load_model_pipeline()

class SymptomRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: SymptomRequest):
    processed = preprocess(req.text)
    raw = pipe(processed, top_k=5)
    predictions = decode_predictions(raw)
    triage = get_triage(predictions, processed)

    return {
        "input_original": req.text,
        "input_processed": processed,
        "predictions": predictions,
        "top_label": predictions[0]["label"],
        "top_confidence_percent": predictions[0]["confidence_percent"],
        "top_description": predictions[0]["description"],
        "triage": triage
    }
