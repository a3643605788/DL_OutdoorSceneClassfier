![Deployment Status](https://github.com/a3643605788/DL_OutdoorSceneClassfier/actions/workflows/deploy.yml/badge.svg)

# Outdoor Scene Image Classification: An E2E MLOps Pipeline

## Executive Summary
This project demonstrates a transition from a standard deep learning model to a **production-ready MLOps ecosystem**. Beyond just achieving high accuracy, it focuses on **deployment efficiency, model interpretability, and cloud scalability**.

By applying Software Engineering best practices, I reduced the container footprint by **81% (8.11GB to 1.5GB)** and implemented a fully automated **CI/CD pipeline** that serves real-world traffic via an asynchronous API.

**Live Demo Status**: The production API is live and hosted on GCP Cloud Run. To align with FinOps (Cloud Financial Management) best practices and optimize resource consumption, the public endpoint is shared upon request.

---

## Tech Stack

* Deep Learning: PyTorch, Torchvision
* Serving: FastAPI, Uvicorn, Pydantic
* MLOps/DevOps: GitHub Actions (CI/CD), Docker, Weights & Biases, GCP (Cloud Run, Artifact Registry)
* Analysis: Scikit-learn, Matplotlib, Grad-CAM

---

## The Four-Phase Evolution

### **Phase 1: Automated Training Pipeline**
* **Objective**: Replace manual experimentation with a reproducible pipeline.
* **Key Deliverables**:
    * **Robust Training**: Integrated **Early Stopping** and **Model Checkpointing** to prevent overfitting and automatically save the `best_resnet.pth` based on validation loss.
    * **Experiment Tracking**: Leveraged **Weights & Biases (W&B)** to monitor hyperparameter sweeps and hardware utilization.

### **Phase 2: Model Evolution & Deep Diagnosis (XAI)**
* **Objective**: Upgrade architecture and explain "Black Box" decisions.
* **Key Deliverables**:
    * **Transfer Learning**: Upgraded from a CNN Baseline (82.3% Acc) to **ResNet18**, achieving **95.0% Accuracy**.
    * **Grad-CAM Interpretability**: Visualized attention maps to diagnose misclassifications. Discovered that "Street" images were often confused with "Buildings" due to architectural texture over-indexing.

### **Phase 3: MLOps Service Layer (SE-to-ML Strength)**
* **Objective**: Wrap the model in a high-performance software interface.
* **Key Deliverables**:
    * **Async FastAPI**: Implemented `async def` endpoints to maximize I/O throughput.
    * **Pydantic Validation**: Strict schema validation ensuring API resilience against malformed data.
    * **Unit Testing**: Comprehensive test suites for inference logic to ensure zero regressions.

### **Phase 4: Containerization & CI/CD Deployment**
* **Objective**: Global accessibility with full automation and "FinOps" (Cost Efficiency).
* **Key Deliverables**:
    * **Multi-stage Docker Build**: Optimized image size from **8.11 GB down to 1.5 GB**, reducing deployment latency and storage costs.
    * **CI/CD Pipeline**: Fully automated deployment via **GitHub Actions**. Every push to the `main` branch triggers an automated build, push to Artifact Registry, and deployment to Cloud Run.
    * **Serverless Infrastructure**: Hosted on **GCP Cloud Run** with strict resource constraints (`--max-instances 3`) to balance availability and budget.

---

## Performance Benchmark

| Metric | CNN Baseline | ResNet18 (Production) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 82.3% | **95.0%** | +12.7% |
| **F1-Score (Weighted)** | 0.81 | **0.95** | +0.14 |
| **Image Footprint** | 8.11 GB | **1.5 GB** | **-81% Size** |
| **Deployment** | Manual | **Auto CI/CD** | **Fully Automated** |

---

## Model Diagnostics (Grad-CAM)
Using **Grad-CAM**, I analyzed why the model focuses on specific features. For the "Buildings" class, the heatmaps clearly show activation on window patterns and rooflines, validating that the model has learned structural features rather than just background colors.

![Grad-CAM Diagnosis](outputs/resnet_diagnosis_gradcam.png)

---

## System Architecture
```text
DL_OutdoorSceneClassfier/
├── .github/workflows/   # CI/CD: Automated deployment scripts
├── api/                 # FastAPI Implementation (Async + Pydantic)
├── src/                 # Core ML Logic (Model Architecture, Grad-CAM)
├── tools/               # Engineering Utilities (Smoke Tests, Summary)
├── data/                # Data Management (Raw/Processed Segregation)
├── outputs/             # Versioned Artifacts (.pth, Metrics, Logs)
├── Dockerfile           # Multi-stage build (Production Optimized)
└── train.py             # E2E Training Entry Point with W&B Integration