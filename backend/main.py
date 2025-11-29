from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model_loader import PlantModel
import os
import uvicorn

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
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.pth"),
)
model = None

@app.on_event("startup")
async def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = PlantModel(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/")
def read_root():
    return {"message": "River Hyacinth Monitor API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
            
        return {
            "prediction": prediction,
            "status": status,
            "alert": alert
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
