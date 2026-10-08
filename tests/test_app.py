import joblib
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import LABELS, MODEL_DIR, PredictRequest, PredictResponse, app

client = TestClient(app)


def test_input_must_be_non_empty_string() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_model_loads() -> None:
    classifier = joblib.load(MODEL_DIR / "classifier.joblib")
    assert hasattr(classifier, "predict")


@pytest.mark.parametrize("text", ["I love it", "I hate it", "It is Thursday"])
def test_inference(text: str) -> None:
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.json()["prediction"] in LABELS.values()


def test_response_is_valid_json() -> None:
    response = client.post("/predict", json={"text": "Great lecture"})
    PredictResponse.model_validate(response.json())


def test_invalid_input_returns_json_error() -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    assert "detail" in response.json()
