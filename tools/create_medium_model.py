import torch
import torch.nn as nn
import os

print("Creating MobileNet-like medium model...")

class SimpleMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Depthwise separable convolutions (like MobileNet)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6()
        
        # Depthwise conv
        self.depthwise = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pointwise conv
        self.pointwise = nn.Conv2d(32, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.depthwise(x)))
        x = self.relu(self.bn3(self.pointwise(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleMobileNet(num_classes=10)
model.eval()

# Smaller input for faster testing
dummy_input = torch.randn(1, 3, 128, 128)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "medium_mobilenet.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    verbose=False
)

print(f"\n✓ Created medium_mobilenet.onnx")
print(f"  Size: {os.path.getsize('medium_mobilenet.onnx') / (1024*1024):.2f} MB")
print(f"  Opset: 13")
print(f"  Input: [1, 3, 128, 128]")
print(f"  Output: [1, 10]")

# Verify
import onnx
onnx_model = onnx.load("medium_mobilenet.onnx")
onnx.checker.check_model(onnx_model)

# Check operations
ops = set(node.op_type for node in onnx_model.graph.node)
print(f"  Operations: {', '.join(sorted(ops))}")
print("✓ Model is valid")
