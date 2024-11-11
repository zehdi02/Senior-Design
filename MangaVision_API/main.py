from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

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

# Load YOLO model 
model = YOLO("yolov8n.pt") 

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
