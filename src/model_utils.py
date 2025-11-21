from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import hf_hub_download
import torch
import json
import re

# ----------------------------------------------------
# LOAD MODEL FROM HUGGINGFACE (NOT LOCAL PATH)
# ----------------------------------------------------
MODEL_REPO = "mayurkumarg/HealSync-Symptom-Model"   # <-- your HF repo

# ----------------------------------------------------
# DOWNLOAD EXTRA FILES FROM HUGGINGFACE
# ----------------------------------------------------

LABEL_MAP_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model/label_map.json"
)

DISEASE_INFO_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model/disease_info.json"
)


# Load JSON files
with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAP = json.load(f)

with open(DISEASE_INFO_PATH, "r") as f:
    DISEASE_INFO = json.load(f)

# ----------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    corrections = {
        "feavar": "fever",
        "caugh": "cough",
        "caigh": "cough",
        "couph": "cough",
        "hedache": "headache",
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    return text


def expand_keywords(text):
    mappings = {
        "tummy pain": "abdominal pain",
        "stomach ache": "abdominal pain",
        "breath problem": "shortness of breath",
    }
    for k, v in mappings.items():
        text = text.replace(k, v)
    return text


def preprocess(text):
    return expand_keywords(clean_text(text))

# ----------------------------------------------------
# LOAD MODEL + TOKENIZER FROM HUGGINGFACE
# ----------------------------------------------------


def load_model_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        subfolder="model"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO,
        subfolder="model"
    )

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )


# ----------------------------------------------------
# FORMAT PREDICTIONS
# ----------------------------------------------------

def decode_predictions(preds):
    final_output = []
    for p in preds:
        idx = p["label"].split("_")[1]
        disease = LABEL_MAP[idx]
        description = DISEASE_INFO.get(disease, "No description available")

        final_output.append({
            "label": disease,
            "confidence_percent": round(p["score"] * 100, 2),
            "description": description
        })
    return final_output

# ----------------------------------------------------
# TRIAGE SYSTEM
# ----------------------------------------------------

TRIAGE_COLORS = {
    "RED": {"color": "#FF0000", "meaning": "Urgent medical attention required"},
    "YELLOW": {"color": "#FFC300", "meaning": "Moderate concern, monitor symptoms"},
    "GREEN": {"color": "#00CC66", "meaning": "Low concern"}
}

SEVERITY_MAP = {
    "Malaria": "RED",
    "Dengue": "RED",
    "Heart attack": "RED",
    "Typhoid": "RED",
    "Pneumonia": "RED",
    "Common Cold": "GREEN",
    "Fungal infection": "GREEN",
    "Allergy": "GREEN"
}

def get_triage(preds, text=None):
    top = preds[0]
    disease = top["label"]
    confidence = top["confidence_percent"] / 100

    if confidence < 0.60:
        level = "YELLOW"
    else:
        level = SEVERITY_MAP.get(disease, "YELLOW")

    return {
        "level": level,
        "color": TRIAGE_COLORS[level]["color"],
        "meaning": TRIAGE_COLORS[level]["meaning"]
    }
