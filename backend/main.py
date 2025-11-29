import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from model_loader import PlantModel
import os
import uvicorn
import gdown
from supabase import Client, create_client
from pydantic import BaseModel
import resend

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
SUPABASE_SETTINGS_TABLE = os.getenv("SUPABASE_SETTINGS_TABLE", "settings")
AUTHORITY_NUMBER_KEY = "authority_number"
AUTHORITY_EMAIL_KEY = "authority_email"
supabase_client: Optional[Client] = None
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
ALERT_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL")
EMAIL_ALERTS_ENABLED = bool(RESEND_API_KEY and ALERT_FROM_EMAIL)

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase client initialised; prediction logs will be stored.")
    except Exception as supabase_error:
        supabase_client = None
        print(f"Failed to init Supabase client: {supabase_error}")
else:
    print("Supabase credentials missing; logging disabled.")

if EMAIL_ALERTS_ENABLED:
    resend.api_key = RESEND_API_KEY
    print("Authority email alerts enabled via Resend; warnings will trigger emails.")
else:
    print("Authority email alerts disabled; set RESEND_API_KEY and RESEND_FROM_EMAIL to enable.")


class AuthorityContactPayload(BaseModel):
    authority_number: str
    authority_email: Optional[str] = None


class AuthorityEmailPayload(BaseModel):
    authority_email: Optional[str] = None


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


async def send_authority_email(metadata: dict):
    if not EMAIL_ALERTS_ENABLED:
        return

    authority_email = metadata.get("authority_email")
    if not authority_email:
        return

    subject = f"River Hyacinth Alert: {metadata.get('prediction', 'Unknown')}"
    body = (
        "Hello,\n\n"
        "A new river hyacinth analysis triggered a warning.\n\n"
        f"Prediction: {metadata.get('prediction', 'Unknown')}\n"
        f"Status: {metadata.get('status', 'Unknown')}\n"
        f"Source: {metadata.get('source', 'unspecified')}\n"
        f"Filename: {metadata.get('filename') or 'N/A'}\n"
        f"Captured At (UTC): {metadata.get('logged_at', datetime.utcnow().isoformat())}\n"
        "\nPlease coordinate clearing operations as soon as possible.\n"
        "\n— River Hyacinth Monitor"
    )

    payload = {
        "from": ALERT_FROM_EMAIL,
        "to": [authority_email],
        "subject": subject,
        "text": body,
    }

    def _send_email():
        resend.Emails.send(payload)

    try:
        await asyncio.to_thread(_send_email)
    except Exception as email_error:
        print(f"Authority email notification failed: {email_error}")


def normalize_authority_number(raw_number: str) -> str:
    digits = "".join(ch for ch in raw_number if ch.isdigit())
    if not digits:
        raise ValueError("Authority number is required.")

    if digits.startswith("63"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]

    if len(digits) != 10:
        raise ValueError("Authority number must be a Philippine number (+63 followed by 10 digits).")

    return f"+63{digits}"


def normalize_authority_email(raw_email: str) -> str:
    email = (raw_email or "").strip().lower()
    if not email:
        raise ValueError("Authority email is required.")

    if "@" not in email:
        raise ValueError("Authority email must include @.")

    local_part, _, domain = email.partition("@")
    if not local_part or not domain or "." not in domain:
        raise ValueError("Authority email must be a valid email address.")

    return email


async def fetch_setting_value(setting_key: str) -> Optional[str]:
    if not supabase_client:
        return None

    def _fetch():
        response = (
            supabase_client
            .table(SUPABASE_SETTINGS_TABLE)
            .select("value")
            .eq("key", setting_key)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return None
        return rows[0].get("value")

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as supabase_error:
        raise HTTPException(status_code=500, detail=f"Failed to fetch {setting_key.replace('_', ' ')}: {supabase_error}") from supabase_error


async def upsert_setting_value(setting_key: str, value: str) -> None:
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Supabase logging is not configured")

    def _upsert():
        supabase_client.table(SUPABASE_SETTINGS_TABLE).upsert({
            "key": setting_key,
            "value": value,
        }).execute()

    try:
        await asyncio.to_thread(_upsert)
    except Exception as supabase_error:
        raise HTTPException(status_code=500, detail=f"Failed to save {setting_key.replace('_', ' ')}: {supabase_error}") from supabase_error


async def delete_setting_value(setting_key: str) -> None:
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Supabase logging is not configured")

    def _delete():
        supabase_client.table(SUPABASE_SETTINGS_TABLE).delete().eq("key", setting_key).execute()

    try:
        await asyncio.to_thread(_delete)
    except Exception as supabase_error:
        raise HTTPException(status_code=500, detail=f"Failed to delete {setting_key.replace('_', ' ')}: {supabase_error}") from supabase_error


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    source: str = "upload",
    authority_number: Optional[str] = Form(None),
    authority_email: Optional[str] = Form(None),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        prediction = model.predict(contents)

        normalized_authority = None
        if authority_number:
            try:
                normalized_authority = normalize_authority_number(authority_number)
            except ValueError as validation_error:
                raise HTTPException(status_code=400, detail=str(validation_error)) from validation_error

        normalized_email = None
        if authority_email:
            try:
                normalized_email = normalize_authority_email(authority_email)
            except ValueError as validation_error:
                raise HTTPException(status_code=400, detail=str(validation_error)) from validation_error
        
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
            "authority_number": normalized_authority,
            "authority_email": normalized_email,
        }
        asyncio.create_task(log_prediction(metadata))

        if alert and normalized_email and EMAIL_ALERTS_ENABLED:
            asyncio.create_task(send_authority_email(metadata))

        return {
            "prediction": prediction,
            "status": status,
            "alert": alert
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/settings/authority-number")
async def get_authority_number_setting():
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Supabase logging is not configured")

    authority_number = await fetch_setting_value(AUTHORITY_NUMBER_KEY)
    authority_email = await fetch_setting_value(AUTHORITY_EMAIL_KEY)
    return {"authority_number": authority_number, "authority_email": authority_email}


@app.post("/settings/authority-number")
async def set_authority_number_setting(payload: AuthorityContactPayload):
    try:
        normalized_number = normalize_authority_number(payload.authority_number)
    except ValueError as validation_error:
        raise HTTPException(status_code=400, detail=str(validation_error)) from validation_error

    normalized_email: Optional[str] = None
    if payload.authority_email is not None:
        trimmed_email = payload.authority_email.strip()
        if trimmed_email:
            try:
                normalized_email = normalize_authority_email(trimmed_email)
            except ValueError as validation_error:
                raise HTTPException(status_code=400, detail=str(validation_error)) from validation_error
            await upsert_setting_value(AUTHORITY_EMAIL_KEY, normalized_email)
        else:
            await delete_setting_value(AUTHORITY_EMAIL_KEY)
    else:
        normalized_email = await fetch_setting_value(AUTHORITY_EMAIL_KEY)

    await upsert_setting_value(AUTHORITY_NUMBER_KEY, normalized_number)
    return {"authority_number": normalized_number, "authority_email": normalized_email}


@app.get("/settings/authority-email")
async def get_authority_email_setting():
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Supabase logging is not configured")

    authority_email = await fetch_setting_value(AUTHORITY_EMAIL_KEY)
    return {"authority_email": authority_email}


@app.post("/settings/authority-email")
async def set_authority_email_setting(payload: AuthorityEmailPayload):
    raw_email = (payload.authority_email or "").strip()
    if not raw_email:
        await delete_setting_value(AUTHORITY_EMAIL_KEY)
        return {"authority_email": None}

    try:
        normalized = normalize_authority_email(raw_email)
    except ValueError as validation_error:
        raise HTTPException(status_code=400, detail=str(validation_error)) from validation_error

    await upsert_setting_value(AUTHORITY_EMAIL_KEY, normalized)
    return {"authority_email": normalized}


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
