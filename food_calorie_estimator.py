import streamlit as st
import cv2
import numpy as np
import xgboost as xgb
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your custom-trained food model if available

# Define food items and nutritional info
food_nutrition = {
    "apple": [100, 25, 0.5, 0.3],
    "banana": [150, 30, 1.2, 0.5],
    "pizza": [200, 35, 2.0, 1.0],
    "burger": [250, 40, 12, 15],
    "rice": [150, 35, 3.5, 0.3],
    "chicken breast": [165, 0, 31, 3.6],
    "salmon": [208, 0, 20, 13],
    "broccoli": [55, 10, 4.5, 0.5],
    "carrot": [41, 10, 0.9, 0.2],
    "potato": [130, 30, 3, 0.1],
    "egg": [70, 1, 6, 5],
    "cheese": [110, 1, 7, 9],
    "donut": [103, 12, 8, 2.5],
    "bread": [80, 15, 3, 1],
    "pasta": [180, 38, 6, 1.5],
    "avocado": [240, 12, 3, 22],
    "nuts": [600, 20, 18, 50],
    "chocolate": [250, 30, 2, 15],
    "yogurt": [150, 20, 8, 4],
    "ice cream": [210, 24, 4, 12],
    "soda": [150, 39, 0, 0],
    "chicken curry": [250, 20, 22, 14],
    "beef steak": [271, 0, 25, 19],
    "tofu": [94, 2, 10, 5],
    "lentils": [230, 40, 18, 0.8],
    "beans": [220, 45, 14, 0.5],
    "butter": [717, 0, 1, 81],
    "hot dog": [884, 0, 0, 100],
    "cake": [304, 82, 0.3, 0],
    "sandwich": [588, 20, 25, 50]
}

# Train XGBoost model
y_train = np.array([
    52, 89, 266, 295, 130, 165, 208, 55, 41, 130, 70, 110, 190, 80, 180, 240,
    600, 546, 150, 207, 150, 250, 271, 94, 230, 220, 717, 290, 257, 450
])
X_train = np.array(list(food_nutrition.values()))
calorie_model = xgb.XGBRegressor()
calorie_model.fit(X_train, y_train)

def predict_calories(food_item):
    if food_item.lower() in food_nutrition:
        features = np.array([food_nutrition[food_item.lower()]])
        return round(calorie_model.predict(features)[0], 2)
    return "Unknown"

def detect_and_estimate(frame):
    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id].lower()

            if label not in food_nutrition:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            calories = predict_calories(label)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 70, 0), 2)
            cv2.putText(frame, f"{label}: {calories} kcal", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
    return frame

# Streamlit UI
st.set_page_config(page_title="Food Calorie Estimator", layout="wide")
st.title("🍽 Food Detection and Calorie Estimator")
st.markdown("Upload a photo or use your webcam to detect food items and estimate their calorie content.")

use_webcam = st.toggle("Use Webcam")
uploaded_file = st.file_uploader("Or Upload a Food Image", type=["jpg", "jpeg", "png"])

if use_webcam:
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
        frame = detect_and_estimate(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    cap.release()

elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    result_img = detect_and_estimate(img)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_img, caption="Detected Food Items with Calorie Estimation", use_column_width=True)



