import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_loader import PlantModel
import os
import uvicorn
import gdown
from supabase import Client, create_client

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model path (prefers env override so deployments can mount models elsewhere)
DEFAULT_MODEL_URL = "https://drive.google.com/uc?id=1_7GG84XkQ0-gX-VYEyOnoWVf3RBTM6mj"
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.pth"),
)
MODEL_URL = os.getenv("MODEL_URL", DEFAULT_MODEL_URL)
model = None

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "prediction_logs")
supabase_client: Optional[Client] = None

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase client initialised; prediction logs will be stored.")
    except Exception as supabase_error:
        supabase_client = None
        print(f"Failed to init Supabase client: {supabase_error}")
else:
    print("Supabase credentials missing; logging disabled.")


def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return

    if not MODEL_URL:
        print(
            "MODEL_URL is not set and the model file is missing. "
            "Set MODEL_URL to a downloadable location or upload the file."
        )
        return

    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading model from {MODEL_URL} to {MODEL_PATH}")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@app.on_event("startup")
async def startup_event():
    global model
    try:
        ensure_model_file()
    except Exception as download_error:
        print(f"Error downloading model: {download_error}")

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = PlantModel(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/")
def read_root():
    return {"message": "River Hyacinth Monitor API is running"}

async def log_prediction(metadata: dict):
    if not supabase_client:
        return

    def _insert():
        supabase_client.table(SUPABASE_TABLE).insert(metadata).execute()

    try:
        await asyncio.to_thread(_insert)
    except Exception as supabase_error:
        print(f"Supabase logging failed: {supabase_error}")


@app.post("/predict")
async def predict(file: UploadFile = File(...), source: str = "upload"):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        prediction = model.predict(contents)
        
        # Determine status based on prediction
        status = "Normal"
        alert = False
        if prediction in ["Moderate Growth", "Large Growth"]:
            status = "Warning"
            alert = True
        metadata = {
            "prediction": prediction,
            "status": status,
            "alert": alert,
            "source": source,
            "filename": file.filename,
            "model_path": MODEL_PATH,
            "file_size": len(contents),
            "logged_at": datetime.utcnow().isoformat(),
        }
        asyncio.create_task(log_prediction(metadata))

        return {
            "prediction": prediction,
            "status": status,
            "alert": alert
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(limit: int = 20, source: Optional[str] = None):
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Supabase logging is not configured")

    limit = max(1, min(limit, 100))

    def _fetch():
        query = (
            supabase_client
            .table(SUPABASE_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
        )
        if source and source.lower() != "all":
            query = query.eq("source", source)
        return query.execute()

    try:
        response = await asyncio.to_thread(_fetch)
    except Exception as supabase_error:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {supabase_error}") from supabase_error

    return {"entries": response.data or []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
