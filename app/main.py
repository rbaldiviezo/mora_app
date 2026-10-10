"""FastAPI backend: trains the real model from a CSV and predicts on demand."""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app import pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR / "data" / "registro_clases_asistencia.csv"
MAX_CSV_BYTES = 2_000_000   # upload size limit (2 MB)

app = FastAPI(title="Riesgo de mora e inasistencia")
state = {"bundle": None}    # trained model kept in memory (single user, simple on purpose)


class PredictRequest(BaseModel):
    student: str = ""
    date: str = ""


def get_bundle():
    """Return the trained model; train with the default CSV the first time."""
    if state["bundle"] is None:
        if not DEFAULT_CSV.exists():
            raise pipeline.ApiError(409, "Aún no hay un modelo entrenado. Carga un CSV primero.")
        state["bundle"] = pipeline.train(DEFAULT_CSV.read_text(encoding="utf-8-sig"))
    return state["bundle"]


@app.exception_handler(pipeline.ApiError)
async def api_error_handler(request: Request, error: pipeline.ApiError):
    return JSONResponse(status_code=error.status_code,
                        content={"error": error.message, "suggestions": error.suggestions})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, error: RequestValidationError):
    return JSONResponse(status_code=422,
                        content={"error": "La solicitud no tiene el formato esperado.", "suggestions": []})


@app.get("/api/status")
def status():
    return pipeline.summary(get_bundle())


@app.post("/api/train")
async def train_model(request: Request):
    """Receives the CSV as plain text in the request body and retrains the model."""
    body = await request.body()
    if len(body) > MAX_CSV_BYTES:
        raise pipeline.ApiError(413, "El archivo supera el límite de 2 MB.")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise pipeline.ApiError(400, "El archivo debe estar codificado en UTF-8.")
    state["bundle"] = pipeline.train(text)
    return pipeline.summary(state["bundle"])


@app.post("/api/predict")
def predict(payload: PredictRequest):
    return pipeline.predict(get_bundle(), payload.student, payload.date)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
