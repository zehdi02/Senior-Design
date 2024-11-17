from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from manga_ocr import MangaOcr
from pydantic import BaseModel
from PIL import Image

from typing import Optional
from datetime import datetime

import numpy as np
import base64
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
    ocr_model: Optional[MangaOcr] = None
    training_results: Optional[dict] = None
    evaluation_results: Optional[dict] = None
    last_trained: Optional[datetime] = None
    uptime: Optional[datetime] = None
    metrics: Optional[dict] = None

mangavision_model = "yolov8n_MangaVision.pt"
model_state = ModelState()  # our model
ocr_state = ModelState()  # manga-ocr API

# init model on startup
@app.on_event("startup")
async def load_model():
    try:
        model_state.model = YOLO(mangavision_model)
        model_state.uptime = datetime.now()
        print(f"{mangavision_model} loaded successfully.")
    except Exception as e:
        print(f"Error loading {mangavision_model}: {str(e)}")
        raise RuntimeError(f"{mangavision_model} loading failed!")
    try:
        ocr_state.ocr_model = MangaOcr()
        ocr_state.uptime = datetime.now()
        print(f"MangaOcr loaded successfully.")
    except Exception as e:
        print(f"Error loading MangaOcr: {str(e)}")
        raise RuntimeError(f"MangaOcr loading failed!")

# close model on shutdown
@app.on_event("shutdown")
async def close_model():
    model_state.model = None
    print(f"{mangavision_model} unloaded.")
    ocr_state.model = None
    print(f"MangaOcr unloaded.")

class TrainingParams(BaseModel):
  data: str
  epochs: int = 100
  batch: int = -1
  imgsz: int = 640
  save: bool = True
  device: str = 'cpu'
  amp: bool = True
  verbose: bool = False
  dropout: float = 0.0
  val: bool = True
  plots: bool = False
  workers: int = 8
  optimizer: bool = 'auto'

class EvaluationParam(BaseModel):
    data: str

@app.post("/train_model")
async def start_training(params: TrainingParams):
    """Endpoint to start training our MangaVision model with given parameters."""
    if not model_state.model:
        raise HTTPException(status_code=500, detail=f"{mangavision_model} is not loaded.")
    try:
        print(f"Starting MangaVision training with parameters: {params}")
        # start training 
        model_state.model.to(params.device)
        model_state.training_results = model_state.model.train(
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
          workers=params.workers,
          optimizer=params.optimizer
          )
        model_state.last_trained = datetime.now()
        return { "message": "Training was successful!" }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/evaluate")
async def evaluate_model(params: EvaluationParam):
    """Endpoint to evaluate our model after training."""
    if not model_state.model:
      raise HTTPException(status_code=500, detail=f"{mangavision_model} is not loaded.")
    try:
      model_state.evaluation_results = model_state.model.val(data=params.data)
      evaluation_metrics = {
          "results_dict": model_state.evaluation_results.results_dict,
          "ap": model_state.evaluation_results.box.ap.tolist(),
          "ap50": model_state.evaluation_results.box.ap50.tolist(),
          "f1": model_state.evaluation_results.box.f1.tolist(),
          "map": model_state.evaluation_results.box.map,
          "map50": model_state.evaluation_results.box.map50,
          "map75": model_state.evaluation_results.box.map75,
          "maps_thres": model_state.evaluation_results.box.maps.tolist(),
          "mean_precision": model_state.evaluation_results.box.mp,
          "mean_recall": model_state.evaluation_results.box.mr,
          "precision": model_state.evaluation_results.box.p.tolist(),
          "recall": model_state.evaluation_results.box.r.tolist(),
          "speed": model_state.evaluation_results.speed,
          "cm": model_state.evaluation_results.confusion_matrix.matrix.tolist(),
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

        # do prediction via 'yolo_prediction' function from 'sort_panel_text_boxes.py' file
        result, width, height = yolo_prediction(image, model_state.model)

        # get sorted bb and conf of panels and text boxes
        sorted_text_boxes_list, sorted_panels_list, \
            sorted_text_boxes_conf_list, sorted_panels_conf_list = sorting_pipeline(image, 1, result, width, height)  
        
        # perform ocr
        extracted_text = generate_transcript(image_bytes, sorted_text_boxes_list, 1, ocr_state.ocr_model)
        
        # get image (in bytes) with bounding boxes drawn
        sorted_image_bytes = draw_sorted_bounding_boxes(image_bytes, \
            sorted_panels_list, sorted_text_boxes_list, sorted_panels_conf_list, sorted_text_boxes_conf_list)

        # convert image bytes to base64
        encoded_image = base64.b64encode(sorted_image_bytes).decode('utf-8')

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
        response_data = {
            "annotations": annotations,
            "mangavision": {
                "text_boxes_sorted": [
                    {
                        "annotation": {
                            'class_id': bb[0],
                            'x_center': bb[1],
                            'y_center': bb[2],
                            'width': bb[3],
                            'height': bb[4],
                        },
                        "confidence": conf,
                        "extracted_text": text
                    } for bb, conf, text in zip(
                        sorted_text_boxes_list, sorted_text_boxes_conf_list, extracted_text
                    )
                ],
                "panels_sorted": [
                    {
                        "annotation": {
                            'class_id': bb[0],
                            'x_center': bb[1],
                            'y_center': bb[2],
                            'width': bb[3],
                            'height': bb[4],
                        },
                        "confidence": conf
                    } for bb, conf in zip(
                        sorted_panels_list, sorted_panels_conf_list
                    )
                ]
            },
            "image": f"data:image/jpeg;base64,{encoded_image}"
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def get_status():
    """Endpoint to check model status and metadata."""
    if model_state.model:
        current_time = datetime.now()
        uptime = str(current_time - model_state.uptime) if model_state.uptime else "N/A"
        metrics = {
            "ap": model_state.evaluation_results.box.ap.tolist(),
            "map": model_state.evaluation_results.box.map,
            "precision": model_state.evaluation_results.box.p.tolist(),
            "recall": model_state.evaluation_results.box.r.tolist(),
            "f1": model_state.evaluation_results.box.f1.tolist()
        } if model_state.evaluation_results else "N/A"
        response = {
            "status": "Model is ready for use!",
            "model_version": "1.0.0",
            "last_trained": model_state.last_trained.strftime("%Y-%m-%d %H:%M:%S") if model_state.last_trained else "N/A",
            "uptime": uptime,
            "metrics": metrics
        }
    else:
        response = {
            "status": "Model not loaded.",
            "model_version": "1.0.0",
            "last_trained": "N/A",
            "uptime": "N/A",
            "metrics": "N/A"
        }
    return response
