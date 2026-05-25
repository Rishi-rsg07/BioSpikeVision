import torch
import torchvision.models as models

def export_to_onnx():
    # Load your model (using dummy weights for this example)
    model = models.mobilenet_v3_small()
    num_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(num_features, 5) # 5 Wildlife classes
    model.eval()

    # Create dummy input matching your preprocessing (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to highly optimized ONNX format
    torch.onnx.export(
        model, 
        dummy_input, 
        "models/biospike_model.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("[SUCCESS] Model converted to models/biospike_model.onnx")

if __name__ == "__main__":
    export_to_onnx()