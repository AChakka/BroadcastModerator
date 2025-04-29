import torch
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  
    model.train(data=r'C:\Users\adith\Downloads\Middle Finger Detection.v3i.yolov8\data.yaml', epochs=50, imgsz=640)

