import torch
import torch.nn as nn
import torchvision.models as models

class DroneDetector(nn.Module):
    """
    CNN-based drone detection model
    Uses transfer learning with ResNet
    """
    def __init__(self, num_classes=2):  # drone vs no-drone
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class DroneDetectionPipeline:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DroneDetector().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
    
    def detect(self, image):
        """
        Detect drone in image
        Returns: (is_drone, confidence, bbox)
        """
        with torch.no_grad():
            # Preprocess image
            img_tensor = self._preprocess(image)
            
            # Forward pass
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            
            is_drone = probs[0, 1] > 0.5
            confidence = probs[0, 1].item()
            
            return is_drone, confidence, None  # bbox can be added with object detection
    
    def _preprocess(self, image):
        # Add preprocessing logic
        pass
