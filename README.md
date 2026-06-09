# Tennis Matchpoint Analytics

Computer vision based tennis match analysis system that detects players, tracks the tennis ball, extracts court keypoints, and generates match analytics such as player speed, shot speed, shot count, mini-court visualization, and heatmap-based tactical insights.

The project is designed to turn raw tennis broadcast footage into structured visual analysis that can help coaches, analysts, and fans understand player movement, court coverage, shot placement, and match intensity.

## Project Highlights

- Detects tennis players using YOLOv8.
- Detects and tracks the tennis ball using a fine-tuned YOLO model.
- Extracts tennis court keypoints using a trained PyTorch model.
- Maps player and ball positions onto a top-down mini-court.
- Calculates shot speed, player speed, average shot speed, and average player speed.
- Annotates output video with player boxes, ball boxes, court lines, frame number, mini-court, and match statistics.
- Supports heatmap-style analytics for player movement and ball placement density.

## Demo Output
The output video includes:

- Player detection bounding boxes
- Ball detection bounding box
- Court keypoint overlay
- Mini-court tracking view
- Player and shot speed dashboard
- Frame counter



## Why This Project Matters

Tennis analysis is traditionally done using expensive tracking systems such as Hawk-Eye or manual match tagging. This project demonstrates how modern computer vision models can approximate key parts of that workflow using broadcast video.

For recruiters, this project shows practical experience with:

- Object detection
- Model fine-tuning
- Keypoint detection
- Video processing
- Sports analytics
- Coordinate transformation
- Data smoothing and interpolation
- Python-based computer vision pipelines

For developers, the project provides a modular starting point for building richer tennis analytics features such as rally classification, player positioning reports, bounce detection, shot-type recognition, and tactical dashboards.

## Tech Stack

- Python 3.8+
- OpenCV
- NumPy
- Pandas
- PyTorch
- Ultralytics YOLO
- YOLOv8 for player detection
- Fine-tuned YOLO model for tennis ball detection
- Custom PyTorch model for court keypoint extraction

## Models Used

### Player Detection

Uses YOLOv8 to detect players in each video frame.

```text
yolov8x
```

### Tennis Ball Detection

Uses a fine-tuned YOLO model trained specifically for tennis ball detection.

```text
models/yolo5_last.pt
```

### Court Keypoint Detection

Uses a PyTorch model trained to identify tennis court keypoints.

```text
models/keypoints_model.pth
```

## Project Structure

```text
tennis_analysis-main/
├── analysis/
│   └── ball_analysis.ipynb
├── constants/
│   └── __init__.py
├── court_line_detector/
│   └── court_line_detector.py
├── input_videos/
│   └── input_video.mp4
├── mini_court/
│   └── mini_court.py
├── models/
│   ├── keypoints_model.pth
│   ├── yolo5_last.pt
│   └── put_models_here.txt
├── output_videos/
│   ├── output_video.avi
│   └── screenshot.jpeg
├── tracker_stubs/
│   ├── ball_detections.pkl
│   └── player_detections.pkl
├── trackers/
│   ├── ball_tracker.py
│   └── player_tracker.py
├── training/
│   ├── tennis_ball_detector_training.ipynb
│   └── tennis_court_keypoints_training.ipynb
├── utils/
│   ├── bbox_utils.py
│   ├── conversions.py
│   ├── player_stats_drawer_utils.py
│   └── video_utils.py
├── main.py
└── yolo_inference.py
```

## How It Works

1. Read the input tennis video.
2. Detect players in every frame using YOLOv8.
3. Detect the tennis ball using a fine-tuned YOLO model.
4. Interpolate missing ball positions to smooth tracking gaps.
5. Detect court keypoints from the first frame.
6. Select the two active tennis players on court.
7. Convert player and ball positions into mini-court coordinates.
8. Detect likely ball shot frames based on trajectory changes.
9. Calculate shot speed and opponent movement speed.
10. Draw all analytics overlays onto the output video.
11. Save the final annotated video.

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install ultralytics torch pandas numpy opencv-python
```

If you are using Apple Silicon or a specific CUDA environment, install the PyTorch version recommended for your machine from the official PyTorch installation guide.

## Required Files

Place the input video here:

```text
input_videos/input_video.mp4
```

Place trained models here:

```text
models/keypoints_model.pth
models/yolo5_last.pt
```

The project can also use detection stubs to avoid rerunning expensive model inference:

```text
tracker_stubs/player_detections.pkl
tracker_stubs/ball_detections.pkl
```

## Running The Project

From the project root:

```bash
python main.py
```

The annotated video will be saved to:

```text
output_videos/output_video.avi
```

## Current Features

### Player Detection

Detects and tracks the active players on court using YOLO-based person detection.

### Ball Detection

Detects the tennis ball frame by frame using a fine-tuned model and interpolates missing positions for smoother analysis.

### Court Keypoint Detection

Identifies tennis court reference points and uses them for visual overlays and coordinate conversion.

### Mini-Court Visualization

Projects player and ball positions onto a top-down mini-court overlay, making movement easier to interpret.

### Speed Analytics

Calculates:

- Last shot speed
- Average shot speed
- Last player speed
- Average player speed
- Number of shots per player

### Heatmap Analytics

The heatmap extension generates a separate image that visualizes:

- Player movement density
- Ball placement density
- Low-to-high activity intensity using blue, green, yellow, and red coloring
- Court layout beneath the heatmap overlay

This helps identify player positioning tendencies, court coverage, preferred ball locations, and tactical patterns.

## Training

The repository includes notebooks for model training:

```text
training/tennis_ball_detector_training.ipynb
training/tennis_court_keypoints_training.ipynb
```

The ball detector is trained using YOLO on tennis ball data. The court keypoint model is trained using PyTorch to predict key court reference points.

## Limitations

- Ball tracking accuracy depends heavily on video quality, camera angle, motion blur, and ball visibility.
- Current shot detection is based on ball trajectory changes and may require tuning for different videos.
- True ball bounce detection is not included by default.
- Player identification can be affected by occlusion, camera cuts, or non-player people near the court.
- The system is best suited for broadcast-style tennis footage with a stable court view.

## Future Improvements

- True ball landing and bounce detection
- Shot classification, such as forehand, backhand, serve, volley, and smash
- Rally segmentation
- Player-specific tactical reports
- Web dashboard for interactive match review
- CSV or JSON export of frame-level analytics
- Match summary generation
- Serve placement maps
- Unforced error and winner classification

## Recruiter Summary

This project demonstrates an end-to-end applied machine learning workflow for sports analytics. It combines object detection, keypoint detection, coordinate transformation, video processing, data interpolation, and visual analytics into a working tennis match analysis system.

It is a strong example of practical AI engineering because it goes beyond model inference and converts raw predictions into meaningful domain-specific insights.

## Developer Notes

- `main.py` orchestrates the full pipeline.
- `trackers/player_tracker.py` handles player detection and filtering.
- `trackers/ball_tracker.py` handles ball detection, interpolation, and shot-frame detection.
- `court_line_detector/court_line_detector.py` predicts and draws court keypoints.
- `mini_court/mini_court.py` converts real frame detections into mini-court coordinates.
- `utils/` contains drawing, geometry, conversion, and video helpers.
- Detection stubs are useful for faster iteration while developing UI or analytics features.