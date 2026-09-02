"""
Trail Camera Object Detection Pipeline
========================================

Single-file conversion of the original Jupyter notebook. Handles:
  1. Dataset download from Roboflow
  2. YOLO model training
  3. Model validation
  4. Video inference + object tracking (with CSV export per video)

Usage examples
--------------
Run the whole pipeline end-to-end:
    python bird_complete.py --all --roboflow-api-key YOUR_KEY

Run individual stages:
    python bird_complete.py --download --roboflow-api-key YOUR_KEY
    python bird_complete.py --train --epochs 150 --model yolo12s.pt
    python bird_complete.py --validate --run 3
    python bird_complete.py --track --run 3

Notes
-----
- Requires an NVIDIA GPU + CUDA-compatible PyTorch build for reasonable
  training/inference speed (run `nvidia-smi` to check your GPU/CUDA version,
  then install the matching torch/torchvision build from
  https://pytorch.org/get-started/locally/).
- Videos to be processed must be placed in a `videos/` folder inside the
  working directory (`--home`, default: current directory). Only `.mp4`
  files are picked up by default (configurable via --video-ext).
- Results CSVs are written to a `results/` folder inside the working
  directory.
"""

import argparse
import os
import sys

import cv2
import pandas as pd


# --------------------------------------------------------------------------- #
# Environment check
# --------------------------------------------------------------------------- #
def check_environment():
    """Print GPU/CUDA/Ultralytics diagnostic info, checking for compatibility."""
    print("=== nvidia-smi ===")
    exit_code = os.system("nvidia-smi")
    if exit_code != 0:
        print("nvidia-smi not found or failed - no NVIDIA GPU detected, "
              "or drivers are not installed.")

    print("\n=== ultralytics.checks() ===")
    import ultralytics
    ultralytics.checks()


# --------------------------------------------------------------------------- #
# Dataset download (Roboflow)
# --------------------------------------------------------------------------- #
def download_dataset(api_key: str, workspace: str, project_name: str,
                      version_number: int, model_format: str = "yolov12"):
    """Download a dataset from Roboflow and return the local dataset object."""
    from roboflow import Roboflow

    if not api_key:
        raise ValueError(
            "A Roboflow API key is required. Pass --roboflow-api-key or set "
            "the ROBOFLOW_API_KEY environment variable."
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(model_format)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(data_yaml: str, model_weights: str = "yolo12n.pt",
                 epochs: int = 100, device=0, plots: bool = True, **train_kwargs):
    """Train a YOLO model on the given dataset. Returns the trained model
    and the training results object.

    Additional keyword args are forwarded to `model.train(...)` - see
    https://docs.ultralytics.com/modes/train/#train-settings for the full
    list of supported hyperparameters.
    """
    from ultralytics import YOLO

    model = YOLO(model_weights)
    results = model.train(data=data_yaml, epochs=epochs, device=device,
                           plots=plots, **train_kwargs)
    return model, results


def load_trained_model(home: str, run: str) -> "YOLO":
    """Load the best weights from a specific training run directory,
    e.g. runs/detect/train3/weights/best.pt.
    """
    from ultralytics import YOLO

    weights_path = os.path.join(home, "runs", "detect", f"train{run}",
                                 "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights found at {weights_path}")

    print(f"Loading model weights from: {weights_path}")
    return YOLO(weights_path)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_model(model, data_yaml: str):
    """Run validation for the given model against the dataset yaml."""
    metrics = model.val(data=data_yaml)
    return metrics


# --------------------------------------------------------------------------- #
# Video inference + tracking
# --------------------------------------------------------------------------- #
def track_videos(model, home: str, video_ext: str = ".mp4",
                  video_dir: str = None, results_dir: str = None,
                  save_annotated: bool = False, classes=None):
    """
    Run detection/tracking over every video in `video_dir` (default:
    <home>/videos) and write one results CSV per video into `results_dir`
    (default: <home>/results).

    Parameters
    ----------
    model : ultralytics.YOLO
        A loaded (trained) YOLO model.
    home : str
        Base working directory.
    video_ext : str
        File extension to filter for (default '.mp4').
    save_annotated : bool
        If True, saves annotated output video via model.track(..., save=True).
    classes : list[int] or None
        Optional list of class indices to restrict tracking to
        (e.g. omit a "feeder" class with classes=[0,2,3,4,5,6]).
    """
    video_dir = video_dir or os.path.join(home, "videos")
    results_dir = results_dir or os.path.join(home, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    videos = os.listdir(video_dir)
    if not videos:
        print(f"No files found in {video_dir}")
        return

    for video in videos:
        if not video.endswith(video_ext):
            print(f"{video} is not a valid video file ({video_ext})")
            continue

        source = os.path.join(video_dir, video)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open video: {source}")
            continue

        data = []
        frame_count = 0

        print(f"Processing {video} ...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            track_kwargs = {"persist": True}
            if save_annotated:
                track_kwargs["save"] = True
            if classes is not None:
                track_kwargs["classes"] = classes

            results = model.track(frame, **track_kwargs)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for i in range(len(result.boxes)):
                        box_id = (int(result.boxes.id[i])
                                  if result.boxes.id is not None else None)
                        cls_val = (int(result.boxes.cls[i])
                                   if result.boxes.cls is not None else None)
                        conf_val = (float(result.boxes.conf[i])
                                    if result.boxes.conf is not None else None)
                        xyxy = result.boxes.xyxy[i]

                        data.append({
                            "frame": frame_count,
                            "id": box_id,
                            "class": model.names[cls_val] if cls_val is not None else None,
                            "confidence": conf_val,
                            "x1": int(xyxy[0]),
                            "y1": int(xyxy[1]),
                            "x2": int(xyxy[2]),
                            "y2": int(xyxy[3]),
                        })

        cap.release()

        df = pd.DataFrame(data)
        out_csv = os.path.join(results_dir, f"{video}_results.csv")
        df.to_csv(out_csv, index=False, header=True)
        print(f"  -> {len(df)} detections saved to {out_csv}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trail camera YOLO pipeline: download, train, validate, track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage selectors
    stage = p.add_argument_group("Pipeline stages (choose one or more)")
    stage.add_argument("--all", action="store_true",
                        help="Run every stage in order: check, download, train, validate, track.")
    stage.add_argument("--check", action="store_true",
                        help="Run GPU/CUDA/ultralytics environment checks.")
    stage.add_argument("--download", action="store_true",
                        help="Download the dataset from Roboflow.")
    stage.add_argument("--train", action="store_true",
                        help="Train a YOLO model.")
    stage.add_argument("--validate", action="store_true",
                        help="Validate a trained model.")
    stage.add_argument("--track", action="store_true",
                        help="Run video inference/tracking and export CSVs.")

    # General
    general = p.add_argument_group("General")
    general.add_argument("--home", default=os.getcwd(),
                          help="Base working directory (defaults to current directory).")

    # Roboflow / dataset
    rf = p.add_argument_group("Dataset (Roboflow)")
    rf.add_argument("--roboflow-api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
                     help="Roboflow API key (or set ROBOFLOW_API_KEY env var).")
    rf.add_argument("--workspace", default="trail-camera",
                     help="Roboflow workspace name.")
    rf.add_argument("--project", default="sam-barrett",
                     help="Roboflow project name.")
    rf.add_argument("--dataset-version", type=int, default=10,
                     help="Roboflow dataset version number.")
    rf.add_argument("--dataset-format", default="yolov12",
                     help="Dataset export format (e.g. yolov12, yolov8, yolov5).")
    rf.add_argument("--data-yaml",
                     help="Path to an existing data.yaml, to skip downloading "
                          "and reuse an already-downloaded dataset.")

    # Training
    tr = p.add_argument_group("Training")
    tr.add_argument("--model", default="yolo12n.pt",
                     help="Base model weights to start training from "
                          "(e.g. yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo11m.pt, yolo10m.pt).")
    tr.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    tr.add_argument("--device", default=0, help="Training device (e.g. 0 for GPU 0, 'cpu').")
    tr.add_argument("--no-plots", action="store_true", help="Disable training plot generation.")

    # Validation / tracking model selection
    run = p.add_argument_group("Model selection (for validate/track)")
    run.add_argument("--run", default="3",
                      help="Which runs/detect/train<run>/weights/best.pt to load "
                           "for validation or tracking.")

    # Tracking
    tk = p.add_argument_group("Video inference / tracking")
    tk.add_argument("--video-dir", default=None,
                     help="Directory containing input videos (default: <home>/videos).")
    tk.add_argument("--results-dir", default=None,
                     help="Directory to write result CSVs (default: <home>/results).")
    tk.add_argument("--video-ext", default=".mp4",
                     help="Video file extension to process.")
    tk.add_argument("--save-annotated", action="store_true",
                     help="Also save annotated output videos.")
    tk.add_argument("--exclude-classes", default=None,
                     help="Comma-separated list of class indices to EXCLUDE from tracking "
                          "(e.g. '1' to omit a feeder class). Applied as the inverse of "
                          "Ultralytics' 'classes' filter isn't native, so this passes the "
                          "remaining classes if --num-classes is also given.")
    tk.add_argument("--classes", default=None,
                     help="Comma-separated list of class indices to INCLUDE in tracking "
                          "(passed straight to model.track(classes=...)).")

    return p


def parse_classes_arg(classes_str):
    if not classes_str:
        return None
    return [int(c.strip()) for c in classes_str.split(",") if c.strip() != ""]


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # If no stage flags given at all, default to --all for convenience.
    any_stage = any([args.all, args.check, args.download, args.train,
                      args.validate, args.track])
    if not any_stage:
        print("No stage selected - defaulting to --all. "
              "Use -h to see individual stage flags.")
        args.all = True

    home = args.home
    dataset_location = None
    model = None

    if args.all or args.check:
        check_environment()

    if args.all or args.download:
        dataset = download_dataset(
            api_key=args.roboflow_api_key,
            workspace=args.workspace,
            project_name=args.project,
            version_number=args.dataset_version,
            model_format=args.dataset_format,
        )
        dataset_location = dataset.location

    # Resolve data.yaml path for train/validate stages
    data_yaml = args.data_yaml
    if data_yaml is None and dataset_location is not None:
        data_yaml = os.path.join(dataset_location, "data.yaml")

    if args.all or args.train:
        if data_yaml is None:
            print("ERROR: --train requires a dataset. Either run --download first "
                  "or pass --data-yaml /path/to/data.yaml", file=sys.stderr)
            sys.exit(1)
        model, _ = train_model(
            data_yaml=data_yaml,
            model_weights=args.model,
            epochs=args.epochs,
            device=args.device,
            plots=not args.no_plots,
        )

    if args.all or args.validate:
        if model is None:
            model = load_trained_model(home, args.run)
        if data_yaml is None:
            print("ERROR: --validate requires a dataset. Either run --download first "
                  "or pass --data-yaml /path/to/data.yaml", file=sys.stderr)
            sys.exit(1)
        validate_model(model, data_yaml)

    if args.all or args.track:
        if model is None:
            model = load_trained_model(home, args.run)
        classes = parse_classes_arg(args.classes)
        track_videos(
            model=model,
            home=home,
            video_ext=args.video_ext,
            video_dir=args.video_dir,
            results_dir=args.results_dir,
            save_annotated=args.save_annotated,
            classes=classes,
        )
        print("Class name mapping:", model.names)


if __name__ == "__main__":
    main()
