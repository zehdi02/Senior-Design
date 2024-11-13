from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import os
from PIL import Image
import io
import numpy as np

# environment variable to avoid library error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# training parameters model
class TrainingParams(BaseModel):
    data: str
    epochs: int = 100
    batch: int = -1
    imgsz: int = 640
    save: bool = True
    device: str
    amp: bool = True
    verbose: bool = False
    dropout: float = 0.0
    val: bool = True
    plots: bool = False

@app.post("/train_model")
async def start_training(params: TrainingParams):
    """Endpoint to start training the YOLO model with given parameters."""
    try:
        print(f"Starting YOLOv8 training with parameters: {params}")

        # init YOLO model
        model = YOLO("yolov8n_MangaVision.pt").to(params.device)

        # train model 
        print('Commence YOLOv8 training...')
        result = model.train(
            data=params.data,
            epochs=params.epochs,
            batch=params.batch,
            imgsz=params.imgsz,
            save=params.save,
            device=params.device,
            verbose=params.verbose,
            amp=params.amp,
            dropout=params.dropout,
            val=params.val,
            plots=params.plots,
        )

        # return success message and results
        return {"message": "Training was successful!", "results": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image bytes
    image_bytes = await file.read()
    
    # Convert the image bytes to a PIL image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
    
    # Convert the PIL image to a format YOLO can accept (numpy array)
    model = YOLO("yolov8n_MangaVision.pt")
    result = model(image)

    # get bounding boxes, class names, and confidence scores
    boxes = result[0].boxes.xyxy  
    class_ids = result[0].boxes.cls  
    confidences = result[0].boxes.conf  

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

@app.get("/")
async def model_init():
    """Endpoint to check if MangaVision model exists."""
    try:
        if os.path.exists("yolov8n_MangaVision.pt"):
            return {"status": "Model is ready for use!", "model_file": "yolov8n_MangaVision.pt"}
        else:
            return {"status": "Missing MangaVision model. 'yolov8n_MangaVision.pt' is required!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model status: {str(e)}")
