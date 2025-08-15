# Smart Traffic Signal 🚦

An AI-powered traffic management system designed to dynamically manage congestion in Indian traffic conditions. This project combines object detection and machine learning to optimize traffic signal timings efficiently.

## Features

- **Dynamic Traffic Management:** Designed and implemented an AI system to handle congestion specific to Indian traffic scenarios.
- **Vehicle Detection:** Collected and processed video data using **Faster R-CNN** to detect and classify vehicles, recording their crossing times to create a dataset.
- **Predictive Modeling:** Built a CSV dataset mapping vehicle counts, types, and crossing times, then trained a **Random Forest regression model** to predict crossing durations.
- **Optimized Signal Timing:** Integrated **YOLOv8** as a lightweight secondary check. YOLOv8 runs on the first frame of every second after half of the predicted time to detect early clearance, optimizing signal timing with minimal resource usage.

## Dataset

- Video data of traffic intersections.
- Processed vehicle counts and types along with crossing times.

## Model Details

- **Object Detection:** Faster R-CNN (primary), YOLOv8 (secondary).
- **Regression Model:** Random Forest for predicting vehicle crossing duration.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/Smart_traffic_signal.git
cd Smart_traffic_signal
