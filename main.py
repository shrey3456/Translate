from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import os
import uvicorn

app = FastAPI()

# Root route to avoid 404 errors
@app.get("/")
def root():
    return {"message": "Translation API is live ✅"}

class TranslationRequest(BaseModel):
    text: str
    source_lang: str  # e.g., en
    target_lang: str  # e.g., hi

model_cache = {}

def load_model(src: str, tgt: str):
    key = f"{src}-{tgt}"
    if key in model_cache:
        return model_cache[key]
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model_cache[key] = (tokenizer, model)
    return tokenizer, model

@app.post("/translate")
def translate_text(req: TranslationRequest):
    try:
        tokenizer, model = load_model(req.source_lang, req.target_lang)
        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"translated_text": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Use dynamic port from environment (important for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default 8000 if PORT not set
    uvicorn.run("main:app", host="0.0.0.0", port=port)
