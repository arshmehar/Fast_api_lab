# FastAPI Diabetes Prediction API

A small API that takes 8 health measurements for a patient and predicts whether they're likely to be diabetic. It's built with FastAPI and serves a decision tree trained on the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle.

I started from the [FastAPI lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) in Ramin Mohammadi's MLOps course, which serves an Iris flower classifier. The point was to learn FastAPI, so instead of only swapping the dataset, I used each change to try a different part of the framework: validation, response models, dependencies and testing.

## What I changed

**A real dataset instead of Iris.** The original lab uses Iris, which comes built into scikit-learn. I switched to the Pima Indians Diabetes data from Kaggle: 768 patients, 8 measurements each, and a yes/no diabetes label. It's loaded from a CSV in `data/`, which is closer to how a real project gets its data.

**Input validation.** The original only checked types, so a negative blood pressure or an age of 500 went straight to the model. Each field now has a sensible range (glucose 0-300, age 1-120 and so on) using Pydantic's `Field`, and FastAPI rejects anything outside it with a 422 before the model sees it. I also added a description for every field and an example patient, so the `/docs` page explains each input and comes pre-filled.

**Responses you can read.** The old response was `{"response": 0}`, which means nothing unless you know the class numbering. Now you get `{"prediction": 1, "label": "diabetic", "confidence": 0.68}`, where confidence is the probability the model gave to its answer.

**Batch predictions.** `POST /predict/batch` takes up to 100 patients in one request and returns the results in the same order. The cap of 100 is there so a single request can't make the server work through an unlimited list. If one patient in the batch is invalid, the error points to exactly which one and which field.

**An API key.** The two prediction endpoints now need an `X-API-Key` header. This uses FastAPI's dependency injection: one small function checks the key, and attaching it with `Depends` protects both endpoints without repeating any code. The health check stays public. The key is compared with `secrets.compare_digest`, which takes the same time however many characters match, so nobody can work out the key by timing the responses.

**Tests.** The original lab had none, so I added six using FastAPI's `TestClient`. The most useful one sends 100 real rows from the CSV through the API and checks the answers match the model's own predictions. If the request fields ever got out of order with the CSV columns, the model would still return predictions, just wrong ones, and this test is what catches that.

**Smaller fixes.** File paths are now based on where each script lives, so you can run things from any folder; the original only worked from inside `src/`. The run command is also corrected: the original README said `uvicorn app:main` instead of `uvicorn main:app`.

## Running it

You'll need Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/train.py          # trains the model and prints test accuracy (about 0.72)
cd src
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs, click **Authorize** and enter `dev-key`. From there you can call any endpoint with **Try it out**. To use your own key, start the server with `API_KEY=your-secret uvicorn main:app --reload`.

## Endpoints

| Method | Path | Needs API key | What it does |
|---|---|---|---|
| GET | `/` | No | Health check, returns `{"status": "healthy"}` |
| POST | `/predict` | Yes | Prediction for one patient |
| POST | `/predict/batch` | Yes | Predictions for 1-100 patients |

## Trying it from the terminal

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

For a batch, wrap the patients in a list:

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

If something's wrong, you get a 401 for a missing or wrong API key, or a 422 for bad input: an out-of-range value, a wrong type, an empty batch or more than 100 patients. The 422 response says which field failed and why.

## Running the tests

Train the model first, then from the project root:

```bash
pytest
```

`TestClient` calls the app directly in Python, so the server doesn't need to be running. The tests cover the health check, rejecting bad API keys, rejecting out-of-range input, the batch size limits, the shape of the response, and the CSV column-order check described above.

## Project layout

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
│   └── main.py               FastAPI app: request/response models, API key check, routes
├── tests/
│   └── test_api.py           API tests with TestClient
├── pytest.ini                lets the tests import from src/
└── requirements.txt
```

## About the data

The [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) comes originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It covers 768 women of Pima Indian heritage, all 21 or older, and 268 of them are diabetic. The API uses snake_case names for the CSV columns:

| CSV column | API field |
|---|---|
| Pregnancies | `pregnancies` |
| Glucose | `glucose` |
| BloodPressure | `blood_pressure` |
| SkinThickness | `skin_thickness` |
| Insulin | `insulin` |
| BMI | `bmi` |
| DiabetesPedigreeFunction | `diabetes_pedigree` |
| Age | `age` |
| Outcome | what the model predicts (1 = diabetic) |

## Known limitations

This is a learning project, so a few things are deliberately simple:

- The model is a depth-3 decision tree with about 72% accuracy. It's fine for practising API work, but not something to make medical decisions with.
- Some columns use 0 to mean "missing" (insulin and skin thickness especially), and I trained on them as they are.
- The model is loaded from disk on every request. Loading it once when the server starts would be the next thing to fix.
- `dev-key` is only a local default. Anything real should set `API_KEY`.

## Credits

- Original lab: [raminmohammadi/MLOps, FastAPI Labs](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) (CC0 1.0)
- Dataset: [UCI Machine Learning on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
