from ultralytics import YOLO
import torch

torch.backends.cudnn.enabled = False

model = YOLO(model='yolov8s.pt',
             task='detect')


# Training.
results = model.train(
   data='datasets/data.yaml',
   imgsz=640,
   epochs=100,
   batch=4,
   name='yolov8s_orangev2_100e'
)