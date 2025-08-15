import cv2
import torch
import csv
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os

# Load YOLOv5 and Faster R-CNN models
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Define vehicle labels for COCO dataset
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'
    #... Add rest of the COCO labels here
]

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

def get_predictions(frame, model, threshold=0.5):
    frame_tensor = F.to_tensor(frame).unsqueeze(0)  # Convert image to tensor
    
    with torch.no_grad():
        predictions = model(frame_tensor)[0]
    
    filtered_predictions = []
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        if score > threshold and 0 <= label < len(COCO_INSTANCE_CATEGORY_NAMES):
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            filtered_predictions.append((box.cpu().numpy(), score.cpu().numpy(), label_name))
    
    return filtered_predictions

def detect_faster_rcnn(frame, model, threshold=0.5):
    predictions = get_predictions(frame, model, threshold)
    
    vehicle_detections = []
    for _, score, label in predictions:
        if label in vehicle_classes:
            vehicle_detections.append(label)
    
    return vehicle_detections

def detect_yolo(frame, model):
    results = model(frame)
    
    vehicle_detections = []
    for detection in results.xyxy[0]:
        class_id = int(detection[5].item())
        if class_id in [2, 3, 5, 7]:  # Vehicle classes
            vehicle_detections.append(model.names[class_id])
    
    return vehicle_detections

def process_normal_video(video_path, csv_log):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    first_7_frames = 7
    vehicles_detected = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

    for i in range(first_7_frames):
        ret, frame = cap.read()
        if not ret:
            break
        vehicle_names = detect_faster_rcnn(frame, faster_rcnn_model)
        for vehicle in vehicle_names:
            if vehicle in vehicles_detected:
                vehicles_detected[vehicle] += 1

    last_30_seconds = fps * 30
    start_frame = max(frame_count - last_30_seconds, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    vehicles_count_log = []
    time_found = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        vehicle_names = detect_yolo(frame, yolo_model)
        vehicles_count_log.append(len(vehicle_names))

    for i, count in enumerate(vehicles_count_log):
        if count < 5:
            time_found = (start_frame + i) / fps
            break
    
    if time_found is None:
        time_found = frame_count / fps

    with open(csv_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([vehicles_detected['motorcycle']//7, vehicles_detected['truck']//7,
                         vehicles_detected['car']//7, vehicles_detected['bus']//7, time_found])

process_normal_video()