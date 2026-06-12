import cv2
import numpy as np
import tensorflow as tf
from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import extract_hybrid_features

def run_demo(model_path='weights/best_hybrid_model.h5'):
    """
    Runs real-time face recognition demo using webcam.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Haar Cascade for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Preprocess the face ROI
            face_roi = frame[y:y+h, x:x+w]
            face_processed = detect_and_align_face(face_roi)
            
            # Extract features
            features = extract_hybrid_features(face_processed)
            
            # Prepare inputs for model
            img_input = face_processed.reshape(1, 128, 128, 1)
            feat_input = features.reshape(1, -1)
            
            # Predict
            prediction = model.predict({'image_input': img_input, 'feature_input': feat_input}, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            label = f"Person {class_id} ({confidence*100:.1f}%)"
            
            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.imshow('Hybrid Face Recognition Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    run_demo()
