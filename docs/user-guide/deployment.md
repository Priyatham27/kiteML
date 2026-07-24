# Production Deployment & Serving

Deploy models into production microservices using FastAPI, ONNX, or Docker containers.

---

## 1. REST API Serving (`kiteml serve`)

```bash
kiteml serve model.pkl --port 8000
```

---

## 2. Exporting to ONNX & Docker

```bash
# ONNX Export
kiteml export model.pkl --format onnx --output model.onnx

# Docker Packaging
kiteml export model.pkl --format docker --output ./docker_deploy/
```
