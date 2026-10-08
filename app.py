from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).resolve().parent / "model"

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

transformer = SentenceTransformer(str(MODEL_DIR / "sentence_transformer.model"))
classifier = joblib.load(MODEL_DIR / "classifier.joblib")


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1)


class PredictResponse(BaseModel):
    prediction: str


app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    embedding = transformer.encode(request.text)
    prediction = classifier.predict([embedding])[0]
    label = LABELS[int(prediction)]
    return PredictResponse(prediction=label)
