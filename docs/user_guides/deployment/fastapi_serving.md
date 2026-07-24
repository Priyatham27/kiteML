# REST API Serving with FastAPI

KiteML's `serving` module (Epic 5) generates production-grade REST APIs using FastAPI, Pydantic, and Uvicorn.

---

## 1. Serving from the CLI (`kiteml serve`)

Serve any trained `.pkl` model artifact or `.kml` pipeline package directly from your terminal:

```bash
kiteml serve model.pkl --port 8000 --workers 4
```

### Auto-Generated Endpoints:
- `POST /predict`: Accepts single or batch JSON records and returns predictions.
- `GET /health`: Health probe returning server status and model version.
- `GET /docs`: Interactive Swagger OpenAPI documentation browser.

---

## 2. Programmatic Model Server

You can also launch or customize the server programmatically in Python:

```python
from kiteml.serving import ModelServer

# Create model server instance
server = ModelServer(model_path="model.pkl")

# Start server on port 8000
server.run(host="0.0.0.0", port=8000)
```
