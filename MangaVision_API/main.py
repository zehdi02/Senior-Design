from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import os
from PIL import Image
import io
import numpy as np
from typing import Optional

# to avoid library error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global variables
class ModelState:
    model: Optional[YOLO] = None
    training_results: Optional[dict] = None

model_state = ModelState()

# init model on startup
@app.on_event("startup")
async def load_model():
    try:
        mangavision_model = "yolov8n_MangaVision.pt"
        model_state.model = YOLO(mangavision_model)
        print(f"{mangavision_model} loaded successfully.")
    except Exception as e:
        print(f"Error loading {mangavision_model}: {str(e)}")
        raise RuntimeError(f"{mangavision_model} loading failed!")

# close model on shutdown
@app.on_event("shutdown")
async def close_model():
    model_state.model = None
    print("Model unloaded.")

# training parameters 
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
    if not model_state.model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        print(f"Starting YOLOv8 training with parameters: {params}")
        # start training 
        model_state.model.to(params.device)
        result = model_state.model.train(
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
        model_state.training_results = result   # store training results to 'model_state.training_results'
        return {"message": "Training was successful!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/evaluate")
async def evaluate_model():
    """Endpoint to evaluate the model after training."""
    if not model_state.training_results:
        raise HTTPException(status_code=404, detail="No training results available.")

    try:
        evaluation_metrics = {
            # model_state.training_results.

            "results_dict": model_state.training_results.results_dict,

            "ap": model_state.training_results.box.ap.tolist(),
            "ap50": model_state.training_results.box.ap50.tolist(),

            "f1": model_state.training_results.box.f1.tolist(),

            "map": model_state.training_results.box.map,
            "map50": model_state.training_results.box.map50,
            "map75": model_state.training_results.box.map75,
            "map_iou_thres": model_state.training_results.box.maps.tolist(),

            "mean_precision": model_state.training_results.box.mp,
            "mean_recall": model_state.training_results.box.mr,

            "precision": model_state.training_results.box.p.tolist(),
            "recall": model_state.training_results.box.r.tolist(),

            "speed": model_state.training_results.speed,

            "cm": model_state.training_results.confusion_matrix.matrix.tolist(),
        }
        return JSONResponse(content=evaluation_metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint to perform prediction on an uploaded image."""
    if not model_state.model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        result = model_state.model(image)

        # extract bounding boxes, class names, and confidence scores
        boxes = result[0].boxes.xyxy
        class_ids = result[0].boxes.cls
        confidences = result[0].boxes.conf

        annotations = []
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            xmin, ymin, xmax, ymax = box.tolist()
            class_name = model_state.model.names[int(class_id)]
            annotations.append({
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax,
                "y_max": ymax,
                "class": class_name,
                "confidence": float(confidence)
            })

        return {"annotations": annotations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def model_init():
    """Endpoint to check if MangaVision model exists."""
    if model_state.model:
        return {"status": "Model is ready for use!"}
    return {"status": "Model not loaded."}
