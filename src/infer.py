from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import json

MODEL_PATH = r"E:/Hackathon/DevHack-MCE/Symptom model/model"
LABEL_MAP_PATH = r"E:/Hackathon/DevHack-MCE/Symptom model/model/label_map.json"

# Load mapping
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

if __name__ == "__main__":
    pipe = load_pipeline()
    text = "I have high fever and joint pain"
    out = pipe(text, top_k=5)

    # Convert LABEL_x → disease name
    for p in out:
        label_num = p["label"].split("_")[1]
        p["label"] = label_map[label_num]

    print(json.dumps(out, indent=2))
