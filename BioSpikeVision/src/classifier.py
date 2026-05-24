import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class WildlifeClassifier:
    def __init__(self, class_labels=None):
        # Default fallback categories for large mammals
        self.labels = class_labels or ["Background", "Bear", "Deer", "Wild Boar", "Wolf"]
        
        # Load ultra-lightweight MobileNetV3
        self.device = torch.device("cpu") # Edge devices typically rely on CPU
        self.model = models.mobilenet_v3_small(pretrained=True)
        
        # Alter the final classification head to match our wildlife classes
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = torch.nn.Linear(num_features, len(self.labels))
        
        # Switch to evaluation mode and apply trace/script optimization
        self.model.eval()
        self.model = torch.jit.script(self.model) # Optimizes runtime execution graph

        # Image preprocessing pipeline optimized for MobileNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, cv2_frame):
        """Runs inference on a standard OpenCV BGR image frame."""
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Transform and add batch dimension
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            _, preds = torch.max(outputs, 1)
            
        return self.labels[preds.item()]