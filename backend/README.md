# Backend Service Prep

This folder contains the FastAPI service that powers River Hyacinth Monitor. Deployments such as Railway only need this directory.

## Requirements
- Python 3.10+
- Install dependencies with `pip install -r requirements.txt`

## Environment Variables
- `MODEL_PATH`: Absolute path to the `best_model.pth` checkpoint. Defaults to the repo root version, but set this in production if the file is stored elsewhere.
- `MODEL_URL`: Optional HTTPS link used to download the checkpoint automatically (defaults to the provided Google Drive file). Set this when storing the model in Drive/S3/etc.

## Running Locally
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Deployment Notes
- Expose the FastAPI `app` via `backend.main:app` (Railway start command example above).
- If the model file is not present, the service will attempt to download it from `MODEL_URL`. Ensure the URL is accessible without manual auth (e.g., Drive sharing link converted to a direct download) and that `MODEL_PATH` points to a writable location like `/tmp/best_model.pth` on Railway.
