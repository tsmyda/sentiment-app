from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


app = FastAPI(title="Sentiment App")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction="positive")
