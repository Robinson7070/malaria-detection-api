from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import base64
import cv2

# Load model
model = tf.keras.models.load_model('cnn_model.h5')

# Get the last conv layer name
last_conv_layer_name = 'conv2d_2'

app = FastAPI(
    title="Malaria Cell Detection API",
    description="Detects malaria parasites in cell images using a CNN achieving 96.52% accuracy. Includes Grad-CAM explainability.",
    version="2.0.0"
)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def generate_gradcam(img_array, model, last_conv_layer_name):
    # Run prediction and get conv layer output directly
    conv_layer = model.get_layer(last_conv_layer_name)
    
    # Create a model that outputs conv layer activations
    feature_extractor = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=conv_layer.output
    )
    
    # Get conv activations
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        conv_outputs = feature_extractor(inputs)
        predictions = model(inputs)
        loss = predictions[:, 0]
    
    # Get gradients with respect to input
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        # Fallback — use raw activations if gradients unavailable
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)[0].numpy()
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
    
    # Normalise
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def overlay_gradcam(original_img, heatmap):
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    original_array = np.array(original_img.resize((128, 128)))
    superimposed = heatmap_colored * 0.4 + original_array * 0.6
    superimposed = np.uint8(superimposed)

    # Convert to base64
    result_img = Image.fromarray(superimposed)
    buffer = io.BytesIO()
    result_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64

@app.get("/")
def home():
    return {
        "model": "Malaria Cell Classifier",
        "accuracy": "96.52%",
        "classes": ["Parasitized", "Uninfected"],
        "status": "running",
        "version": "2.0.0 with Grad-CAM explainability"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()

    # Preprocess
    img_array, original_img = preprocess_image(image_bytes)

    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    # Interpret result
    if confidence >= 0.5:
        label = "Uninfected"
        confidence_score = confidence
    else:
        label = "Parasitized"
        confidence_score = 1 - confidence

    # Generate Grad-CAM
    heatmap = generate_gradcam(img_array, model, last_conv_layer_name)
    gradcam_image = overlay_gradcam(original_img, heatmap)

    return JSONResponse({
        "prediction": label,
        "confidence": round(confidence_score * 100, 2),
        "message": f"Cell is {label} with {round(confidence_score * 100, 2)}% confidence",
        "gradcam_image": gradcam_image,
        "gradcam_note": "Base64 encoded PNG heatmap showing model focus regions"
    })