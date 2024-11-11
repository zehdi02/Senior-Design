from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os

app = FastAPI()

# Configure CORS to allow requests from localhost:8001 (where HTML is served)
origins = [
    "http://127.0.0.1:8001", 
    "http://localhost:8001",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# API Key for authentication (you can modify this to use a better method)
api_key_header = APIKeyHeader(name="Authorization")

# Load YOLO model (initial model)
model = YOLO("yolov8n.pt")

# Define the training function
def train_model(dataset_path: str, hyperparameters: dict):
    if not dataset_path or not os.path.exists(dataset_path):
        raise ValueError("Invalid dataset path")
    
    # Load the model (initial YOLO model)
    model = YOLO("yolov8n.pt")  # Base model
    # Train the model with the provided dataset and hyperparameters
    model.train(data=dataset_path, epochs=hyperparameters.get("epochs", 50), batch=hyperparameters.get("batch_size", 16))
    
    # Return the model after training
    return model

@app.post("/train")
async def train(training_data: dict, api_key: str = Depends(api_key_header)):
    # Basic API key check
    if not api_key or api_key != "mangavision59867":
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        # Extract dataset path and hyperparameters from the request
        dataset_path = training_data.get("dataset_path")
        hyperparameters = training_data.get("hyperparameters", {})

        # Trigger the training
        trained_model = train_model(dataset_path, hyperparameters)

        # Return the success message along with the trained model version
        return {"status": "success", "message": "Training started successfully", "model_version": trained_model.version}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Open image with PIL
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to numpy array for the model
    img_array = np.array(image)

    # Use YOLO model to get predictions (adjust for your actual model)
    model = YOLO("yolov8n_MangaVision.pt")  # Replace with your model path
    result = model(img_array)  # Get the prediction results

    # Access the bounding boxes in xyxy format (list of boxes)
    boxes = result[0].boxes.xyxy  
    class_ids = result[0].boxes.cls  
    confidences = result[0].boxes.conf  

    # Loop through boxes and extract the coordinates, class name, and confidence score
    annotations = []
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        xmin, ymin, xmax, ymax = box.tolist() 
        class_name = model.names[int(class_id)]  
        
        annotations.append({
            "x_min": xmin,
            "y_min": ymin,
            "x_max": xmax,
            "y_max": ymax,
            "class": class_name,
            "confidence": float(confidence)  
        })

    return {"annotations": annotations}
