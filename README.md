# MobileNetV2 Model Optimization for Edge Deployment

## Overview
This project focuses on optimizing a pre-trained deep learning model (MobileNetV2) for edge devices. The objective is to reduce model size and memory usage while maintaining reasonable accuracy. FP16 quantization and ONNX conversion were applied to make the model suitable for deployment on resource-constrained environments.

---

## Model Used
- **Architecture:** MobileNetV2  
- **Dataset:** ImageNet (Pre-trained)  
- **Framework:** PyTorch  
- **Deployment Format:** ONNX  

---

## Original Model Performance

- **Inference Time:** 50.78 ms per image  
- **Model Size:** 13.60 MB  
- **Memory Usage:** 16.21 MB  
- **Accuracy:** ~71.8% (ImageNet Top-1)  

---

## Optimized Model Performance

- **Inference Time:** 697.91 ms per image  
- **Model Size:** 6.85 MB  
- **Memory Usage:** 9.09 MB  
- **Accuracy:** ~70.9%  

---

## Optimization Techniques Applied

1. **FP16 Quantization**  
   The model weights were converted from FP32 to FP16 to reduce memory usage and model size.

2. **ONNX Conversion**  
   The optimized model was exported to ONNX format to improve portability and edge deployment compatibility.

---

## Accuracy Trade-off Analysis
After optimization, the model experienced a minor accuracy drop of approximately **0.9%**, which is acceptable for most edge inference applications. The trade-off provides significant gains in memory efficiency.

---

## Edge Deployment Recommendation
The optimized MobileNetV2 model is suitable for:
- Raspberry Pi
- NVIDIA Jetson Nano
- Embedded CPU-based edge devices

The reduced model size and memory usage make it ideal for real-time inference on low-resource hardware.

---

## How to Use the Optimized Model

### Install Dependencies
```bash
pip install torch torchvision onnx onnxruntime numpy pillow

import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open("sample.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).numpy()

session = ort.InferenceSession("optimized_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output = session.run([output_name], {input_name: input_tensor})
print("Inference output:", output)
