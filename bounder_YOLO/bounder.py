from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    results = model.train(data='bounder.yaml', epochs=1, batch=8, device='cuda', optimizer='Adam', lr0=.5)


if __name__ == '__main__':
    main()