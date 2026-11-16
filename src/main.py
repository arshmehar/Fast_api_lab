import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from predict import predict_data


app = FastAPI()

# shortcut: hard-coded default key for local lab use, set the API_KEY env var before deploying anywhere real
API_KEY = os.getenv("API_KEY", "dev-key")
# auto_error=False so a missing key reaches verify_api_key and gets the same 401 as a wrong key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

LABELS = {0: "not diabetic", 1: "diabetic"}


def verify_api_key(api_key: str | None = Depends(api_key_header)):
    """
    Dependency that rejects any request without the correct X-API-Key header.
    FastAPI runs it before the route function; raising here stops the request.
    """
    # compare_digest takes the same time whether the first or last character is wrong,
    # so attackers can't guess the key by timing responses
    if not api_key or not secrets.compare_digest(api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


class PatientData(BaseModel):
    """
    One patient's measurements. Field() limits are checked by Pydantic before the
    route runs; out-of-range values get an automatic 422 response.
    Field order must match the CSV column order: to_row() passes values to the model in this order.
    """
    pregnancies: int = Field(ge=0, le=20, description="Number of times pregnant")
    glucose: float = Field(ge=0, le=300, description="Plasma glucose concentration (mg/dL)")
    blood_pressure: float = Field(ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(ge=0, le=1000, description="2-hour serum insulin (mu U/ml)")
    bmi: float = Field(ge=0, le=80, description="Body mass index (kg/m^2)")
    diabetes_pedigree: float = Field(ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(ge=1, le=120, description="Age in years")

    # Shown as the pre-filled request body in /docs (first row of the dataset)
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
                "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50,
            }]
        }
    }

    def to_row(self):
        return list(self.model_dump().values())


class PredictionResponse(BaseModel):
    prediction: int = Field(description="1 = diabetic, 0 = not diabetic")
    label: str
    confidence: float = Field(description="Probability the model gives to its predicted class")


class BatchRequest(BaseModel):
    # Capped so a single request can't send an unbounded list
    patients: list[PatientData] = Field(min_length=1, max_length=100)


def run_model(patients):
    """Run the model on a list of PatientData and build one PredictionResponse per patient."""
    predictions, confidences = predict_data([p.to_row() for p in patients])
    return [
        PredictionResponse(prediction=int(p), label=LABELS[int(p)], confidence=float(c))
        for p, c in zip(predictions, confidences)
    ]


"""Modern web apps use a technique named routing. This helps the user remember the URLs. 
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/ 
they’d see /account/1234/."""

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Concurrent (multiple tasks can run simultaneously)"""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_diabetes(patient: PatientData):
    """
    Predict whether one patient is diabetic.
    Requires the X-API-Key header.
    Example:
        POST /predict
        {"pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
         "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50}
        Response:
        {"prediction": 1, "label": "diabetic", "confidence": 0.68}
    """
    try:
        return run_model([patient])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[PredictionResponse], dependencies=[Depends(verify_api_key)])
async def predict_diabetes_batch(batch: BatchRequest):
    """
    Predict for 1 to 100 patients in one request. Results come back in the same order.
    Requires the X-API-Key header.
    Example:
        POST /predict/batch
        {"patients": [{...patient 1...}, {...patient 2...}]}
    """
    try:
        return run_model(batch.patients)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
