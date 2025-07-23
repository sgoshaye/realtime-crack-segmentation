# realtime-crack-segmentation

Real-time pavement crack detection and segmentation using a fine-tuned **YOLOv8** model, with a classic **Canny Edge** pipeline as a baseline. Includes a multithreaded OpenCV system for interactive Canny parameter tuning.

## Video Demonstration

[![Real Time Crack Segmentation Project](https://img.youtube.com/vi/b1YT1ybdzcg/0.jpg)](https://www.youtube.com/watch?v=b1YT1ybdzcg)

## Highlights
- **Model:** YOLOv8n-seg fine-tuned for 30 epochs on an open-source crack dataset + 39 s of my own pavement video.
- **Baseline:** Canny edge detector with real-time sliders (thresholds, blur, morphology) implemented via producer/consumer threads and queues.
- **Result:** YOLOv8 generalizes and detects cracks far better; Canny is noisy and requires constant retuning.

## Dataset
- Roboflow “Crack dataset” (open-source).  
  Ref: University. (2022, December). *Crack dataset*. Roboflow Universe.  
  Link: https://docs.ultralytics.com/datasets/segment/crack-seg/#dataset-yaml

## Code Structure

Custom_Pkg/

Detector.py # VideoProcessor (base), MLDetector, CannyDetector, CannyRealTime

Datasets/crack-seg/ # train/valid/test + data.yaml

runs/segment/train/ # YOLO weights & metrics (Ultralytics output)

video_input/ # raw backyard footage

video_output/ # processed videos (YOLO boxes, Canny contours)

YOLO_train_model.ipynb

detect_cracks.py # main script: run YOLO or Canny pipelines


## Known Limitation
OpenCV trackbar labels: clicking to type values works once per kernel session; subsequent clicks can freeze the GUI. (Dragging sliders is fine.)
