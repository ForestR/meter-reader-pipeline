from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

class DialDetector:
    def __init__(self, config_path: str = "config/train_dial.yaml"):
        """Initialize the dial detector model.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLO(self.config['model']['name'])
        self.device = self.config['training']['device']
        
    def train(self, data_yaml_path: str):
        """Train the dial detector model.
        
        Args:
            data_yaml_path (str): Path to the data configuration YAML file
        """
        self.model.train(
            data=data_yaml_path,
            epochs=self.config['training']['epochs'],
            batch=self.config['training']['batch_size'],
            imgsz=self.config['training']['img_size'],
            device=self.device,
            project=self.config['logging']['save_dir'],
            name='dial_detector'
        )
    
    def predict(self, image_path: str):
        """Predict dial locations in an image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            list: List of detected dials with their bounding boxes
        """
        results = self.model.predict(
            source=image_path,
            conf=0.25,
            device=self.device
        )
        return results[0].boxes.data.cpu().numpy()
    
    def save(self, path: str):
        """Save the model weights.
        
        Args:
            path (str): Path to save the model weights
        """
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str, config_path: str = "config/train_dial.yaml"):
        """Load a trained model.
        
        Args:
            path (str): Path to the saved model weights
            config_path (str): Path to the configuration file
            
        Returns:
            DialDetector: Loaded model instance
        """
        instance = cls(config_path)
        instance.model = YOLO(path)
        return instance 