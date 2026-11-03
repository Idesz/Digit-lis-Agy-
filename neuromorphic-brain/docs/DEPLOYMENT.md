# Deployment Guide: Docker, GPU/CPU & Service Configuration

This guide provides operational instructions for building, running, and deploying the Neuromorphic Brain Emulation platform.

---

## 1. Prerequisites

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **NVIDIA Container Toolkit** (Optional, required for GPU acceleration):
  - CUDA Driver >= 525
  - Tested on NVIDIA CUDA 12.1 runtime

---

## 2. Docker Compose Deployment

The platform runs multi-container services including `brain-core`, `redis`, and `qdrant`.

### Starting the Stack
```bash
cd neuromorphic-brain
docker compose up --build -d
```

### Checking Services Status
```bash
docker compose ps
docker compose logs -f brain-core
```

### Stopping the Stack
```bash
docker compose down
```

---

## 3. CPU vs. GPU Automatic Hardware Detection

The core application automatically probes for NVIDIA GPU availability via PyTorch:
- If `torch.cuda.is_available() == True`: All spiking tensors, surrogate convolutions, and predictive models are initialized on CUDA devices (`cuda:0`).
- If GPU is absent or disabled: The platform falls back seamlessly to multi-threaded CPU execution (`cpu`) without throwing errors.

In `docker-compose.yml`, GPU devices are reserved under `deploy.resources.reservations`. If executing on a non-GPU host with standard docker-compose, remove or comment out the `reservations` section or run natively using Python virtual environments.

---

## 4. Local Python Native Execution (No Docker)

If you prefer to run the system directly in your Python environment:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch training and simulation pipeline
python train.py
```

### In-Memory Fallbacks
When running outside Docker without active Redis (port 6379) or Qdrant (port 6333) instances:
- **Redis Client** falls back to an internal thread-safe in-memory key-value cache.
- **Qdrant Client** falls back to Qdrant's embedded `:memory:` engine or an internal vector store index.
- All 6 pillars execute reliably without connection timeouts.

---

## 5. Verification & Testing

Run unit and integration test suites:
```bash
pytest tests/ -v
```
