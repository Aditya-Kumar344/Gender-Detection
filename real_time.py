import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

MODEL_PATH = "gender_detection_final.h5"
IMG_SIZE = 128

model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
prev_time = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = rgb[y:y+h, x:x+w]
        try:
            resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        except:
            continue

        batch = np.expand_dims(resized / 255.0, axis=0)
        prob = model.predict(batch, verbose=0)[0][0]

        if prob > 0.5:
            gender = "Male"
            confidence = prob * 100
            color = (0, 255, 0)
        else:
            gender = "Female"
            confidence = (1 - prob) * 100
            color = (255, 0, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{gender} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Real-Time Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()