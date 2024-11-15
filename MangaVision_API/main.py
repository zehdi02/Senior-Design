from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pydantic import BaseModel
from PIL import Image
from typing import Optional
from datetime import datetime
import numpy as np
import io
import sys
import os

pipeline_folder = os.path.join(os.path.dirname(__file__), '..', 'MangaVision_pipeline')
sys.path.append(pipeline_folder)

from display_sorted_panels_textboxes import *
from sort_panel_textboxes import *
from mangavision import *

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

class ModelState:
    model: Optional[YOLO] = None
    training_results: Optional[dict] = None
    last_trained: Optional[datetime] = None
    uptime: Optional[datetime] = None
    metrics: Optional[dict] = None

model_state = ModelState()

# init model on startup
@app.on_event("startup")
async def load_model():
    try:
        mangavision_model = "yolov8n_MangaVision.pt"
        model_state.model = YOLO(mangavision_model)
        model_state.uptime = datetime.now()
        print(f"{mangavision_model} loaded successfully.")
    except Exception as e:
        print(f"Error loading {mangavision_model}: {str(e)}")
        raise RuntimeError(f"{mangavision_model} loading failed!")

# close model on shutdown
@app.on_event("shutdown")
async def close_model():
    model_state.model = None
    print("Model unloaded.")

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
        model_state.training_results = result
        model_state.last_trained = datetime.now()
        return {f'"message": "Training was successful!, "parameters": {params}'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/evaluate")
async def evaluate_model():
    """Endpoint to evaluate the model after training."""
    if not model_state.training_results:
        raise HTTPException(status_code=404, detail="No training results available.")

    try:
        evaluation_metrics = {
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

        sorted_text_boxes_list, sorted_panels_list, sorted_text_boxes_conf_list, sorted_panels_conf_list = sorting_pipeline(image)

        extracted_text = generate_transcript(image_bytes, sorted_text_boxes_list)
        # sorted_image = draw_sorted_bounding_boxes(img_fp, sorted_panels_list, sorted_text_boxes_list, sorted_panels_conf_list, sorted_text_boxes_conf_list)
        
        # do prediction
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
                "class": class_name,
                "class_id": int(class_id),
                "confidence": float(confidence),
                "y_max": ymax,
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax
            })

        return {"annotations": annotations, 
                "mangavision": { 
                    'sorted_text_boxes': {
                        'tb_ocr': extracted_text, 
                        'tb_bb': sorted_text_boxes_list,
                        'tb_conf': sorted_text_boxes_conf_list,
                        }, 
                    'sorted_panels': {
                        'p_bb': sorted_panels_list,
                        'p_conf': sorted_panels_conf_list,
                        }
                    }
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def get_status():
    """Enhanced endpoint to check model status and metadata."""
    if model_state.model:
        current_time = datetime.now()
        uptime = str(current_time - model_state.uptime) if model_state.uptime else "N/A"
        
        response = {
            "status": "Model is ready for use!",
            "model_version": "1.0.0",
            "last_trained": model_state.last_trained.strftime("%Y-%m-%d %H:%M:%S") if model_state.last_trained else "N/A",
            "uptime": uptime,
        }
    else:
        response = {
            "status": "Model not loaded.",
            "model_version": "N/A",
            "last_trained": "N/A",
            "uptime": "N/A",
        }
    
    return response

