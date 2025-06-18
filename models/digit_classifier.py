import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from pathlib import Path

class DigitClassifier:
    def __init__(self, config_path: str = "config/train_digit_cls.yaml"):
        """Initialize the digit classifier model.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load pretrained ResNet model
        self.model = models.resnet18(pretrained=self.config['model']['pretrained'])
        
        # Modify the final layer for digit classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.config['model']['num_classes'])
        
        self.device = torch.device(self.config['training']['device'])
        self.model = self.model.to(self.device)
        
    def train(self, train_loader, val_loader, criterion, optimizer):
        """Train the digit classifier model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
        """
        self.model.train()
        for epoch in range(self.config['training']['epochs']):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
            self.model.train()
    
    def predict(self, image):
        """Predict digit class.
        
        Args:
            image (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Predicted class probabilities
        """
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            return torch.softmax(output, dim=1)
    
    def save(self, path: str):
        """Save the model weights.
        
        Args:
            path (str): Path to save the model weights
        """
        torch.save(self.model.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, config_path: str = "config/train_digit_cls.yaml"):
        """Load a trained model.
        
        Args:
            path (str): Path to the saved model weights
            config_path (str): Path to the configuration file
            
        Returns:
            DigitClassifier: Loaded model instance
        """
        instance = cls(config_path)
        instance.model.load_state_dict(torch.load(path))
        return instance 