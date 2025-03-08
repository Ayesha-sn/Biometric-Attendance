import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load face detection model using Haarcascade
face_cascade = cv2.CascadeClassifier(r"C:\Users\acema\OneDrive\Desktop\firebase db\haarcascade_frontalface_default.xml")

# Load mask detection model
mask_model = load_model(r"C:\Users\acema\OneDrive\Desktop\firebase db\mask_detection_model.keras")

# Function for real-time mask detection
def real_time_mask_detection():
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (startX, startY, w, h) in faces:
            endX, endY = startX + w, startY + h
            
            face = frame[startY:endY, startX:endX]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            
            predictions = mask_model.predict(face)[0]
            
            if len(predictions) == 2:
                mask, withoutMask = predictions
            else:
                mask = predictions[0]
                withoutMask = 1 - mask  # Assuming binary classification
            
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{label}: {max(mask, withoutMask) * 100:.2f}%", 
                        (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Real-Time Mask Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start real-time detection
real_time_mask_detection()
