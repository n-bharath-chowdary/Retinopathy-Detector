import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Define constants
IMG_SIZE = (224, 224)
MODEL_PATH = "retinopathy_model.h5"  # Path to trained model
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]  # Severity levels

# Load trained model
model = keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")


# Function to preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Error: Cannot read image at {img_path}")

    img = cv2.resize(img, IMG_SIZE)  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values (0-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to predict retinopathy severity
def predict_retinopathy():
    root = tk.Tk()
    root.withdraw()  # Hide root window
    root.attributes('-topmost', True)  # Keep file dialog on top
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    root.destroy()
    if not img_path:
        print("No image selected. Exiting.")
        return

    try:
        img = preprocess_image(img_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)  # Get class with highest probability
        confidence = np.max(prediction)  # Get confidence score

        # Display result
        print(f"Prediction: {CLASS_NAMES[predicted_class]} (Confidence: {confidence * 100.00:.2f})")

        # Show the image with prediction
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {CLASS_NAMES[predicted_class]} ({confidence * 100.00:.2f})")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


# Run prediction
predict_retinopathy()
