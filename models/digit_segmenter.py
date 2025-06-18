import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path
import segmentation_models_pytorch as smp

class DigitSegmenter:
    def __init__(self, config_path: str = "config/train_digit_seg.yaml"):
        """Initialize the digit segmenter model.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = smp.Unet(
            encoder_name=self.config['model']['encoder_name'],
            encoder_weights=self.config['model']['encoder_weights'],
            in_channels=self.config['model']['in_channels'],
            classes=self.config['model']['classes']
        )
        
        self.device = torch.device(self.config['training']['device'])
        self.model = self.model.to(self.device)
        
    def train(self, train_loader, val_loader, criterion, optimizer):
        """Train the digit segmenter model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
        """
        self.model.train()
        for epoch in range(self.config['training']['epochs']):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
            
            val_loss /= len(val_loader)
            print(f'Validation set: Average loss: {val_loss:.4f}')
            self.model.train()
    
    def predict(self, image):
        """Predict digit segmentation mask.
        
        Args:
            image (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Predicted segmentation mask
        """
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            return torch.sigmoid(output)
    
    def save(self, path: str):
        """Save the model weights.
        
        Args:
            path (str): Path to save the model weights
        """
        torch.save(self.model.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, config_path: str = "config/train_digit_seg.yaml"):
        """Load a trained model.
        
        Args:
            path (str): Path to the saved model weights
            config_path (str): Path to the configuration file
            
        Returns:
            DigitSegmenter: Loaded model instance
        """
        instance = cls(config_path)
        instance.model.load_state_dict(torch.load(path))
        return instance 