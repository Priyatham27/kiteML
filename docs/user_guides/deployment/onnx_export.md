# ONNX Model Graph Export & Runtime Execution

KiteML's `deployment.onnx_export` module converts trained scikit-learn and KiteML pipeline artifacts into optimized **ONNX (Open Neural Network Exchange)** graph format for cross-platform, high-throughput inference.

---

## 1. Exporting via CLI (`kiteml export`)

Convert a trained model `.pkl` to ONNX format from the command line:

```bash
kiteml export model.pkl --format onnx --output model.onnx
```

---

## 2. Programmatic ONNX Export

```python
from kiteml.deployment import ONNXExporter

exporter = ONNXExporter()
onnx_bytes = exporter.export(result.model, sample_input=X_test.iloc[:1])

# Save ONNX model to disk
with open("model.onnx", "wb") as f:
    f.write(onnx_bytes)
```

---

## 3. High-Performance ONNX Runtime Inference

Execute inference using `onnxruntime` without Python machine learning dependencies:

```python
import onnxruntime as ort
import numpy as np

# Create ONNX Inference Session
session = ort.InferenceSession("model.onnx")

# Prepare input tensor
input_name = session.get_inputs()[0].name
input_data = X_test.to_numpy().astype(np.float32)

# Run inference session
outputs = session.run(None, {input_name: input_data})
predictions = outputs[0]
```
