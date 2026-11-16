# FastAPI Diabetes Prediction API

A REST API that predicts whether a patient is diabetic from 8 medical measurements. It serves a decision tree trained on the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle, using FastAPI and uvicorn.

This project extends the [FastAPI lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) from Ramin Mohammadi's MLOps course labs, which served an Iris flower classifier. I swapped in a new dataset and added input validation, richer responses, a batch endpoint, API key authentication and automated tests.

## What I changed from the original lab

| Area | Original lab | This version |
|---|---|---|
| Dataset | Iris, loaded from scikit-learn (4 features, 3 classes) | Pima Indians Diabetes from Kaggle, loaded from CSV (768 patients, 8 features, 2 classes) |
| Input validation | Type checks only | Allowed range for every field (e.g. glucose 0-300) using Pydantic `Field`; out-of-range values return 422 |
| API docs | Empty request body in `/docs` | Example patient pre-filled in `/docs`, plus a description for every field |
| Response | `{"response": 0}` | `{"prediction": 1, "label": "diabetic", "confidence": 0.68}` |
| Endpoints | `GET /`, `POST /predict` | Adds `POST /predict/batch` for 1-100 patients per request |
| Security | None | `/predict` and `/predict/batch` require an `X-API-Key` header, checked by a FastAPI dependency |
| Tests | None | 6 pytest tests using FastAPI's `TestClient` |
| File paths | Relative to the current folder, so scripts only worked from `src/` | Relative to each file, so scripts work from any folder |

## Endpoints

| Method | Path | Needs API key | Description |
|---|---|---|---|
| GET | `/` | No | Health check, returns `{"status": "healthy"}` |
| POST | `/predict` | Yes | Prediction for one patient |
| POST | `/predict/batch` | Yes | Predictions for 1-100 patients, returned in the same order |

Interactive documentation is generated automatically at `/docs`.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/train.py          # trains the model, prints test accuracy (about 0.72)
cd src
uvicorn main:app --reload    # serves the API at http://127.0.0.1:8000
```

The API key defaults to `dev-key` for local use. To set your own, start the server with `API_KEY=your-secret uvicorn main:app --reload`.

In the browser, open http://127.0.0.1:8000/docs, click **Authorize**, enter the key, then use **Try it out** on any endpoint.

## Example requests

Single prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{"pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
       "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50}'
```

```json
{"prediction": 1, "label": "diabetic", "confidence": 0.6833333333333333}
```

Batch prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{"patients": [
        {"pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35, "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50},
        {"pregnancies": 1, "glucose": 85, "blood_pressure": 66, "skin_thickness": 29, "insulin": 0, "bmi": 26.6, "diabetes_pedigree": 0.351, "age": 31}
      ]}'
```

```json
[{"prediction": 1, "label": "diabetic", "confidence": 0.6833333333333333},
 {"prediction": 0, "label": "not diabetic", "confidence": 0.9813084112149533}]
```

Error responses:

| Situation | Status | Example |
|---|---|---|
| Missing or wrong API key | 401 | `{"detail": "Invalid or missing API key"}` |
| Value out of range or wrong type | 422 | `glucose: -5` returns "Input should be greater than or equal to 0" |
| Empty batch or more than 100 patients | 422 | `{"patients": []}` |

## Running the tests

From the project root (train the model first):

```bash
pytest
```

The tests use `TestClient`, which calls the app directly in Python, so the server doesn't need to be running. They cover:

- the health check
- rejection of missing and wrong API keys (401)
- out-of-range input (422)
- batch size limits (422)
- the response format (label matches prediction, confidence between 0.5 and 1)
- API predictions matching the model's own predictions on 100 real rows from the CSV. This catches the request fields getting out of order with the CSV columns, a bug that would give wrong answers without raising an error.

## How it works

**Request validation.** `PatientData` is a Pydantic model with one field per CSV column. FastAPI parses the JSON body into it, converts types, and checks the `Field(ge=..., le=...)` limits before the route function runs. Invalid requests get a 422 response listing each bad field. The fields are declared in the same order as the CSV columns, because that's the order the model expects.

**Response models.** `response_model=PredictionResponse` makes FastAPI serialize the output to that shape and document it in `/docs`. `confidence` is the probability the decision tree assigns to its predicted class (`predict_proba`).

**Batch endpoint.** `BatchRequest` holds a `list[PatientData]` capped at 100 items, so one request can't send an unlimited amount of data. Each patient is validated individually, and errors point to the exact item (for example `["body", "patients", 3, "glucose"]`).

**API key authentication.** `verify_api_key` is a dependency attached with `dependencies=[Depends(verify_api_key)]`. FastAPI runs it before each protected route, and if it raises `HTTPException(401)` the route never runs. `APIKeyHeader` reads the `X-API-Key` header and adds the Authorize button to `/docs`. Keys are compared with `secrets.compare_digest` so response timing doesn't reveal how much of a guessed key was correct.

## Project structure

```
Fast_api_lab
├── data/
│   └── diabetes.csv          Kaggle dataset
├── model/
│   └── diabetes_model.pkl    trained decision tree
├── src/
│   ├── data.py               loads the CSV, train/test split
│   ├── train.py              trains and saves the model
│   ├── predict.py            loads the model, returns predictions and confidence
│   └── main.py               FastAPI app: models, API key dependency, routes
├── tests/
│   └── test_api.py           API tests with TestClient
├── pytest.ini                adds src/ to the import path for tests
└── requirements.txt
```

## Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (originally from the National Institute of Diabetes and Digestive and Kidney Diseases). It covers 768 female patients of Pima Indian heritage, aged 21 or older; 268 are diabetic.

| Column | API field |
|---|---|
| Pregnancies | `pregnancies` |
| Glucose | `glucose` |
| BloodPressure | `blood_pressure` |
| SkinThickness | `skin_thickness` |
| Insulin | `insulin` |
| BMI | `bmi` |
| DiabetesPedigreeFunction | `diabetes_pedigree` |
| Age | `age` |
| Outcome | prediction target (1 = diabetic) |

## Limitations

- The model is a depth-3 decision tree with about 72% test accuracy. It is for learning API development, not medical use.
- Several columns use 0 for missing values (for example insulin and skin thickness), and the model is trained on them as-is.
- The model file is loaded from disk on every request; loading it once at startup would be faster.
- `dev-key` is a local default. Set the `API_KEY` environment variable in any real deployment.

## Credits

- Original lab: [raminmohammadi/MLOps - FastAPI Labs](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) (CC0 1.0)
- Dataset: [UCI Machine Learning / Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
