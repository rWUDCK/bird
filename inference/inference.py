import cv2
import os
import argparse
from ultralytics import YOLO
from pathlib import Path
import time
import torch

class YOLOInference:
    def __init__(self, model_path, conf_threshold=0.001, iou_threshold=0.45, device='auto'):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.agnostic_nms = False
        self.model.max_det = 20
        self.model.to(self.device)
        
        self.class_names = self.model.names
        
        print(f"Model loaded from: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
    
    def process_frame(self, frame):
        results = self.model(frame, device=self.device)
        
        detections = []
        frame_with_detections = frame.copy()
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            print(f"RAW DETECTIONS: {xyxys}, confs: {confs}, clss: {clss}")
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, xyxys[i])
                conf = confs[i]
                cls = clss[i]
                if conf > self.conf_threshold:
                    class_id = int(cls)
                    class_name = self.class_names[class_id]
                    color = (0, 255, 0)
                    cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        frame_with_detections,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        frame_with_detections,
                        label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_name': class_name,
                        'class_id': class_id
                    })
        
        return frame_with_detections, detections
    
    def process_video(self, video_path, output_path=None, show_display=True):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        print("Starting video processing...")
        if show_display:
            print("Press 'q' to quit, 's' to save current frame")
        
        display_available = show_display
        if show_display:
            try:
                cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)
                display_available = True
            except Exception as e:
                print(f"Warning: Could not initialize display window: {e}")
                print("Continuing without display...")
                display_available = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
            
            annotated_frame, detections = self.process_frame(frame)
            
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Detections: {len(detections)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            if detections:
                y_offset = 60
                for i, det in enumerate(detections[:5]):
                    det_text = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(
                        annotated_frame,
                        det_text,
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
            
            if writer:
                writer.write(annotated_frame)
            
            if display_available:
                try:
                    cv2.imshow('YOLO Inference', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        frame_save_path = f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(frame_save_path, annotated_frame)
                        print(f"Saved frame to: {frame_save_path}")
                    elif key == ord('p'):
                        cv2.waitKey(0)
                except Exception as e:
                    print(f"Display error: {e}")
                    display_available = False
            else:
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        if writer:
            writer.release()
        if display_available:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        
        if output_path:
            print(f"Annotated video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Video Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model weights')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated video (optional)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold (default: 0.45)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, or cpu (default: auto)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    yolo_inference = YOLOInference(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    yolo_inference.process_video(
        video_path=args.video,
        output_path=args.output,
        show_display=not args.no_display
    )

if __name__ == "__main__":
    main() 