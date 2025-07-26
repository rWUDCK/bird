# YOLO Video Inference

This folder contains scripts for running YOLO inference on videos with real-time display and video saving capabilities.

## Files

- `inference.py` - Main inference script for processing videos
- `data/` - Folder containing test videos (video1.mp4, video2.mp4, video3.mp4)

## Features

### Real-time Video Processing
- Load and process video files using OpenCV
- Display detections in real-time with bounding boxes and confidence scores
- Show frame counter, FPS, and detection count
- Interactive controls (pause, save frame, quit)

### Video Annotation
- Draw bounding boxes around detected objects
- Display class names and confidence scores
- Save annotated videos with all detections

### Flexible Configuration
- Adjustable confidence and IoU thresholds
- Support for custom trained YOLO models
- Optional real-time display (can run headless)

## Usage

### Basic Usage

```bash
# Process a single video with real-time display
python inference.py --model path/to/model.pt --video path/to/video.mp4

# Save annotated video
python inference.py --model path/to/model.pt --video path/to/video.mp4 --output annotated_video.mp4

# Adjust confidence threshold
python inference.py --model path/to/model.pt --video path/to/video.mp4 --conf 0.5

# Run without display (headless mode)
python inference.py --model path/to/model.pt --video path/to/video.mp4 --no-display
```

### Command Line Arguments

- `--model`: Path to trained YOLO model weights (required)
- `--video`: Path to input video file (required)
- `--output`: Path to save annotated video (optional)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--no-display`: Disable real-time display

### Interactive Controls

When running with display enabled:
- **q**: Quit the application
- **s**: Save current frame as image
- **p**: Pause/unpause video


## Requirements

Make sure you have the following dependencies installed:
- OpenCV (`opencv-python`)
- PyTorch (`torch`)
- NumPy (`numpy`)
