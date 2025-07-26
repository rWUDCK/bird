#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO


class YOLOTrainer:
    
    def __init__(self, workspace_name="trail-camera", project_name="sam-barrett", version_number=10):
        load_dotenv()
        
        self.workspace_name = workspace_name
        self.project_name = project_name
        self.version_number = version_number
        self.api_key = self._get_api_key()
        self.dataset_location = None
        
    def _get_api_key(self):
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please add it to your .env file.")
        return api_key
    
    def download_dataset(self):
        try:
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace_name).project(self.project_name)
            version = project.version(self.version_number)
            dataset = version.download("yolov12")
            
            self.dataset_location = dataset.location
            print("Dataset downloaded successfully!")
            print(f"Dataset path: {self.dataset_location}")
            
            return self.dataset_location
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def train_model(self, model_config=None, epochs=100, batch_size=16, img_size=640, device=0):
        if not self.dataset_location:
            raise ValueError("Dataset not downloaded. Call download_dataset() first.")
        
        data_yaml_path = os.path.join(self.dataset_location, "data.yaml")
        
        model = YOLO("yolo12m.pt")
        
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            device=device,
            plots=True
        )
        
        print(f"Training completed! Results saved to: {results.save_dir}")
        return results
        
    def run_training_pipeline(self, epochs=100, batch_size=16, img_size=640, device=0):
        print("Starting YOLO training pipeline...")
        
        print("Step 1: Downloading dataset...")
        self.download_dataset()
        
        print("Step 2: Training model...")
        results = self.train_model(epochs=epochs, batch_size=batch_size, img_size=img_size, device=device)
        
        print("Training pipeline completed!")
        return results


def main():
    trainer = YOLOTrainer()
    
    results = trainer.run_training_pipeline(epochs=100, batch_size=16, img_size=640, device=0)
    print(f"Training results: {results}")


if __name__ == "__main__":
    main()
