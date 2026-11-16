from fastapi.testclient import TestClient

from data import load_data
from main import API_KEY, app
from predict import predict_data

client = TestClient(app)
AUTH = {"X-API-Key": API_KEY}
# First row of data/diabetes.csv, keys in CSV column order
PATIENT = {"pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
           "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50}


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_missing_or_wrong_api_key_is_rejected():
    assert client.post("/predict", json=PATIENT).status_code == 401
    assert client.post("/predict", json=PATIENT, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/predict/batch", json={"patients": [PATIENT]}).status_code == 401


def test_predict_returns_label_and_confidence():
    response = client.post("/predict", json=PATIENT, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("diabetic" if body["prediction"] == 1 else "not diabetic")
    assert 0.5 <= body["confidence"] <= 1  # two classes, so the predicted one has at least 50%


def test_out_of_range_input_is_rejected():
    assert client.post("/predict", json={**PATIENT, "glucose": -5}, headers=AUTH).status_code == 422
    assert client.post("/predict", json={**PATIENT, "age": 500}, headers=AUTH).status_code == 422


def test_batch_size_limits():
    assert client.post("/predict/batch", json={"patients": []}, headers=AUTH).status_code == 422
    assert client.post("/predict/batch", json={"patients": [PATIENT] * 101}, headers=AUTH).status_code == 422


def test_api_matches_model_on_real_rows():
    # Catches PatientData fields drifting out of CSV column order: the model would still run, just wrongly
    X, _ = load_data()
    rows = X[:100]
    patients = [dict(zip(PATIENT, row.tolist())) for row in rows]
    response = client.post("/predict/batch", json={"patients": patients}, headers=AUTH)
    assert response.status_code == 200
    expected, _ = predict_data(rows)
    assert [p["prediction"] for p in response.json()] == expected.tolist()
