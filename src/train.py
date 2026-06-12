import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import extract_hybrid_features
from src.models.hybrid_model import build_hybrid_model
from src.evaluation.metrics import evaluate_model

def load_orl_dataset(data_path='data/raw/orl'):
    """
    Loads ORL dataset. Expected structure:
    data/raw/orl/
        s1/
            1.pgm, 2.pgm ...
        s2/
            ...
    """
    images = []
    hybrid_features = []
    labels = []
    class_names = []

    if not os.path.exists(data_path):
        print(f"Dataset path {data_path} not found. Please download ORL dataset.")
        return None, None, None, None

    subjects = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    for idx, subject in enumerate(subjects):
        class_names.append(subject)
        subject_path = os.path.join(data_path, subject)
        for img_name in os.listdir(subject_path):
            img_path = os.path.join(subject_path, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Preprocess
            face = detect_and_align_face(img)
            
            # Extract features
            features = extract_hybrid_features(face)
            
            images.append(face.reshape(128, 128, 1))
            hybrid_features.append(features)
            labels.append(idx)
            
    return np.array(images), np.array(hybrid_features), np.array(labels), class_names

def train_hybrid_system(data_path='data/raw/orl'):
    # Load data
    print("Loading and preprocessing dataset...")
    X_img, X_feat, y, class_names = load_orl_dataset(data_path)
    
    if X_img is None:
        return

    # Split data
    X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = \
        train_test_split(X_img, X_feat, y, test_size=0.2, random_state=42, stratify=y)
    
    num_classes = len(class_names)
    feature_dim = X_feat.shape[1]
    
    print(f"Data loaded: {len(X_img)} samples, {num_classes} classes")
    print(f"Feature dimension: {feature_dim}")

    # Build model
    model = build_hybrid_model(img_shape=(128, 128, 1), feature_dim=feature_dim, num_classes=num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint('weights/best_hybrid_model.h5', save_best_only=True)
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        {'image_input': X_img_train, 'feature_input': X_feat_train},
        y_train,
        validation_data=({'image_input': X_img_test, 'feature_input': X_feat_test}, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Evaluate
    print("Evaluating model...")
    evaluate_model(model, {'image_input': X_img_test, 'feature_input': X_feat_test}, y_test, class_names=class_names)
    
    # Save final model
    model.save('weights/final_hybrid_model.h5')
    print("Training complete. Model saved to weights/final_hybrid_model.h5")

if __name__ == "__main__":
    train_hybrid_system()
