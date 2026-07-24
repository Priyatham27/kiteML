# Docker Container Packaging

KiteML's `deployment.docker_export` module package models into production Docker containers complete with a pre-configured `Dockerfile`, Uvicorn REST server, schema validators, and health probes.

---

## 1. Exporting Docker Artifacts (`kiteml export`)

Generate container deployment files:

```bash
kiteml export model.pkl --format docker --output ./docker_deploy/
```

Generated Directory Structure:
```text
docker_deploy/
├── Dockerfile
├── requirements.txt
├── app.py
└── model.pkl
```

---

## 2. Generated Dockerfile Overview

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Building & Running the Container

```bash
cd docker_deploy/

# Build Docker image
docker build -t kiteml-churn-service:v1 .

# Run Docker container
docker run -d -p 8000:8000 kiteml-churn-service:v1
```
