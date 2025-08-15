import cv2
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import joblib

# Load YOLOv5 and Faster R-CNN models
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Define COCO vehicle labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# Function to count vehicles using Faster R-CNN
def count_vehicles_faster_rcnn(frame, model):
    # Convert frame to tensor and normalize
    frame_tensor = F.to_tensor(frame).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(frame_tensor)[0]
    
    vehicle_counts = {'motorcycle': 0, 'truck': 0, 'car': 0, 'bus': 0}

    # Filter out vehicle classes
    for label in predictions['labels']:
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        if label_name in vehicle_counts:
            vehicle_counts[label_name] += 1
    
    # Return counts in the correct order for your model
    return [vehicle_counts['motorcycle'], vehicle_counts['truck'], vehicle_counts['car'], vehicle_counts['bus']]

# Function to detect vehicles using YOLOv5
def detect_vehicles_yolo(frame, model):
    results = model(frame)
    detections = results.xyxy[0]  # Get bounding box predictions

    vehicle_count = 0
    for detection in detections:
        label = int(detection[5])  # Class index
        if COCO_INSTANCE_CATEGORY_NAMES[label] in vehicle_classes:
            vehicle_count += 1
    
    return vehicle_count, detections

# Initialize video capture
cap = cv2.VideoCapture(f'demo_data/1.MOV')
ret, first_frame = cap.read()

# Get vehicle counts from the first frame using Faster R-CNN
vehicle_counts = count_vehicles_faster_rcnn(first_frame, faster_rcnn_model)

# Use your model to predict the time based on the vehicle counts
time_model = joblib.load('vehicle_time_predictor_2_1.pkl')
predicted_time = time_model.predict([vehicle_counts])
predicted_time_duration = predicted_time * 1000  # Convert to milliseconds

# Start timing
start_time = time.time()
yolo_started = False
last_yolo_frame_time = 0  # Keeps track of the last time YOLO was run

current_vehicle_count = sum(vehicle_counts)
yolo_start_time = 0
stop = False
stop_time=0
# Process video frames
# Initialize an orange signal time tracker
orange_signal_triggered = False
orange_signal_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    current_time = (time.time() - start_time) * 1000  # in milliseconds

    # Check if YOLO should start (after 50% of predicted time)
    if not yolo_started and current_time > 0.5 * predicted_time_duration:
        yolo_started = True
        yolo_start_time = current_time / 1000  # in seconds

    vehicle_count_list = []
    if yolo_started and not orange_signal_triggered:
        # Run YOLO only once per second
        if int(current_time // 1000) != int(last_yolo_frame_time // 1000):
            # Detect vehicles using YOLOv5
            current_vehicle_count, detections = detect_vehicles_yolo(frame, yolo_model)
            last_yolo_frame_time = current_time

            # Draw bounding boxes and display vehicle count
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                label = int(cls)
                if COCO_INSTANCE_CATEGORY_NAMES[label] in vehicle_classes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the vehicle count and status on the right side of the window
    cv2.putText(frame, f"Vehicle Count: {current_vehicle_count}", (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"YOLO Started at: {yolo_start_time:.2f}s", (frame.shape[1] - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Predicted Time: {predicted_time[0]:.2f}s", (frame.shape[1] - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # If vehicle count is less than 10, set stop to True
    if current_vehicle_count < 10:
        stop = True
        if stop_time == 0:
            stop_time = time.time() - start_time

    # Handle orange and red signal logic
    if stop:
        if not orange_signal_triggered:
            orange_signal_triggered = True
            orange_signal_start_time = time.time()
        else:
            orange_duration = 3 - (time.time() - orange_signal_start_time)

            # Display "ORANGE" for 5 seconds
            if orange_duration <= 3 and orange_duration > 0:
                cv2.putText(frame, f"ORANGE : {orange_duration:.2f}s", (frame.shape[1] - 300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            else:
                # After 5 seconds, switch to "STOP"
                cv2.putText(frame, f"STOP : {stop_time:.2f}s", (frame.shape[1] - 300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        # Reset orange signal logic if stop is not triggered
        orange_signal_triggered = False
        orange_signal_start_time = 0

    # Display the frame
    cv2.imshow('Vehicle Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
