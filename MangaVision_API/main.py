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

# variable to store training results
training_results = None

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
        # store training results to global variable 'training_results' for later access during evaluate
        result = training_results

        # return success message and results
        return {"message": "Training was successful!", "results": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/evaluate")
async def evaluate_model():
    """Endpoint to evaluate the model after training."""
    global training_results
    if training_results is None:
        raise HTTPException(status_code=404, detail="No training results available. Please train the model first.")

    evaluation_metrics = {}
    
    # detection metrics
    evaluation_metrics["results_dict"] = training_results.results_dict

    # per-class AP 
    per_class_metrics = []
    for i in range(len(training_results.box.ap_class_index)):
        per_class_metrics.append({
            "class_index": i,
            "all_ap": training_results.box.all_ap[i],    # AP scores for all IoU thresholds
            "ap50-95": training_results.box.ap[i],       
            "ap50": training_results.box.ap50[i]         
        })
    evaluation_metrics["per_class_metrics"] = per_class_metrics

    # mean precision & mean recall
    evaluation_metrics["precision"] = training_results.box.mp    
    evaluation_metrics["recall"] = training_results.box.mr       

    # speed 
    evaluation_metrics["speed"] = training_results.speed

    # curves (ROC, PR, F1, etc.)
    evaluation_metrics["curves"] = training_results.curves

    # confusion matrix metrics
    confusion_matrix = {
        "matrix": training_results.confusion_matrix.matrix.tolist(),  # numpy array to list
        "num_classes": training_results.confusion_matrix.nc,
        "confidence_threshold": training_results.confusion_matrix.conf,
        "iou_threshold": training_results.confusion_matrix.iou_thres
    }
    evaluation_metrics["confusion_matrix"] = confusion_matrix

    return JSONResponse(content=evaluation_metrics)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read the uploaded image bytes
    image_bytes = await file.read()
    
    # convert the image bytes to a PIL image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
    
    # convert the PIL image to a format YOLO can accept (numpy array)
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
