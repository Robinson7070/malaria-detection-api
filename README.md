# Malaria Cell Detection API

A production-style REST API for automated malaria parasite 
detection from microscopy images, with Grad-CAM explainability.

**Model accuracy: 96.52% | Built with FastAPI and TensorFlow**

---

## What it does

Send a cell image to the API and receive:
- Prediction — Parasitized or Uninfected
- Confidence score — how certain the model is
- Grad-CAM heatmap — which region of the cell drove the decision

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and status |
| `/health` | GET | Health check |
| `/predict` | POST | Upload image, get prediction + Grad-CAM |

---

## Example Response

```json
{
  "prediction": "Parasitized",
  "confidence": 94.32,
  "message": "Cell is Parasitized with 94.32% confidence",
  "gradcam_image": "base64_encoded_heatmap",
  "gradcam_note": "Base64 encoded PNG heatmap showing model focus regions"
}
```

---

## Why Grad-CAM matters

For medical AI, a prediction alone is not enough. Clinicians need 
to know WHY the model made a decision. Grad-CAM visualises which 
region of the cell the model focused on — confirming it learned 
genuine pathology, not dataset artefacts.

---

## Model

Custom CNN trained on the NIH Malaria Cell Images dataset:
- 27,558 microscopy images
- 2 classes: Parasitized and Uninfected
- Architecture: 3-block CNN (32→64→128 filters)
- Test accuracy: 96.52% | Recall: 97.68%

The model file is not included in this repo due to size.
Train your own using the Kaggle notebook linked below.

**Kaggle notebook:** [Malaria Cell Classification](https://kaggle.com/robinsoncvictor/coursework-malaria-cell-image-classification)

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Add your trained model
# Place cnn_model.h5 in the project root

# Start the API
uvicorn main:app --reload

# View interactive docs
# Open http://127.0.0.1:8000/docs
```

---

## Stack

Python · FastAPI · TensorFlow · Keras · OpenCV · Pillow · NumPy

---

## Project context

Individual coursework for CN7023 Artificial Intelligence & 
Machine Vision, MSc AI & Data Science, University of East 
London, 2026.

Part of a broader portfolio: github.com/Robinson7070
