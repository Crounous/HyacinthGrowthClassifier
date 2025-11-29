# Backend Service Prep

This folder contains the FastAPI service that powers River Hyacinth Monitor. Deployments such as Railway only need this directory.

## Requirements
- Python 3.10+
- Install dependencies with `pip install -r requirements.txt`

## Environment Variables
- `MODEL_PATH`: Absolute path to the `best_model.pth` checkpoint. Defaults to the repo root version, but set this in production if the file is stored elsewhere.

## Running Locally
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Deployment Notes
- Expose the FastAPI `app` via `backend.main:app` (Railway start command example above).
- Upload or mount the model file and set `MODEL_PATH` appropriately before starting the service.
