import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # ✅ Disable Streamlit Watcher

import asyncio
import streamlit as st
import nest_asyncio
# ✅ Fix asyncio error in Python 3.12+
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

# Now import the rest
import cv2
import numpy as np
import torch
import joblib
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights  # ✅ Fix pretrained warning
from ultralytics import YOLO
import pytesseract

# ✅ Fix ResNet50 model loading (removes warning)
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.eval()

# Load YOLO models
truck_model = YOLO("yolov8n.pt")  
license_plate_model = YOLO(r"C:\Users\Lenova\Desktop\new dumper ON\models\license_plate_detector.pt")

# Load Number Plate Classifier
number_plate_model = joblib.load(r"C:\Users\Lenova\Desktop\new dumper ON\models\model.pkl")  

# Image Preprocessing for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image):
    """Extract features from the license plate for classification."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 8")  # Improved OCR text detection
    text_clarity = len(text.strip())

    edges = cv2.Canny(gray, 50, 150)
    edge_sharpness = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    rust_level = 0  # No rust detection in grayscale images
    return [rust_level, text_clarity, edge_sharpness]

# Streamlit UI
st.title("🚛 NEW DUMPER WITH OLD NUMBER DETECTOR")
st.write("Upload an image of a truck to detect its type and check for fraud.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read and display the image
    image = Image.open(uploaded_file)
    image_cv = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Truck classification
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(img_tensor)
    truck_type = "Old" if torch.argmax(output).item() == 0 else "New"

    # Truck detection
    results = truck_model(image_cv)
    number_plate_type = "Unknown"

    for result in results:
        detected_classes = set([int(cls) for cls in result.boxes.cls])
        st.write(f"Detected Classes in Image: {detected_classes}")  # Debugging output

        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            TRUCK_CLASS_ID = 7  # Verify this using detected_classes
            if int(cls) == TRUCK_CLASS_ID:
                x1, y1, x2, y2 = map(int, box[:4])
                truck_crop = image_cv[y1:y2, x1:x2]

                # License plate detection
                lp_results = license_plate_model(truck_crop)
                if not lp_results:
                    st.warning("⚠️ No license plate detected! Please try another image.")
                    continue

                for lp_result in lp_results:
                    for lp_box in lp_result.boxes.xyxy:
                        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box[:4])

                        # Adjust coordinates to original image
                        lp_x1 += x1
                        lp_y1 += y1
                        lp_x2 += x1
                        lp_y2 += y1

                        lp_crop = image_cv[lp_y1:lp_y2, lp_x1:lp_x2]
                        st.image(lp_crop, caption="Detected License Plate", use_container_width=True)

                        # Extract features and classify number plate
                        features = np.array(extract_features(lp_crop)).reshape(1, -1)
                        if np.isnan(features).any():
                            st.error("Error in extracted features: NaN values found.")
                        else:
                            number_plate_prediction = number_plate_model.predict(features)[0]
                            number_plate_type = "New" if number_plate_prediction == 1 else "Old"

    # Display Results
    st.subheader("Results:")
    st.write(f"**Truck Type:** {truck_type}")
    st.write(f"**Number Plate Type:** {number_plate_type}")

    # Fraud detection
    if truck_type == "New" and number_plate_type == "Old":
        st.error("🚨 NEW DUMPER WITH OLD NUMBER DETECTED 🚨")
    else:
        st.success("✅ No fraud detected")
