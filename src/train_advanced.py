import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import extract_hybrid_features
from src.models.hybrid_model import (
    build_hybrid_model, 
    build_cnn_only_model, 
    build_feature_only_model
)
from src.evaluation.metrics import evaluate_model, compute_comprehensive_metrics
import json
from datetime import datetime

class CenterLossLayer(layers.Layer):
    """
    Center Loss implementation for improved intra-class compactness.
    Based on Wen et al. "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """
    def __init__(self, num_classes, feature_dim, alpha=0.5, **kwargs):
        super(CenterLossLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha  # Learning rate for centers
        self.centers = None
    
    def build(self, input_shape):
        # Initialize centers randomly
        self.centers = self.add_weight(
            shape=(self.num_classes, self.feature_dim),
            initializer='uniform',
            trainable=False,
            name='centers'
        )
        super(CenterLossLayer, self).build(input_shape)
    
    def call(self, inputs):
        features, labels = inputs
        
        # Compute center loss
        labels = tf.cast(labels, tf.int32)
        centers_batch = tf.gather(self.centers, labels)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(features - centers_batch), axis=1))
        
        # Update centers (moving average)
        delta = features - centers_batch
        update_centers = tf.scatter_sub(self.centers, labels, self.alpha * delta)
        
        with tf.control_dependencies([update_centers]):
            return tf.identity(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'alpha': self.alpha
        })
        return config


def compute_center_loss(features, labels, centers, alpha=0.5):
    """
    Compute center loss for batch
    """
    labels = tf.cast(labels, tf.int32)
    centers_batch = tf.gather(centers, labels)
    
    # Distance between features and centers
    diff = features - centers_batch
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))
    
    return loss


class MultiStageTrainer:
    """
    Implements the 4-stage training strategy from the research paper:
    Stage 1: Pretrain CNN branch (20 epochs)
    Stage 2: Pretrain feature branch (10 epochs)  
    Stage 3: Joint fine-tuning (30-50 epochs)
    Stage 4: Optional ensemble
    """
    
    def __init__(self, num_classes, feature_dim, img_shape=(128, 128, 1)):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.img_shape = img_shape
        self.history = {
            'stage1': {},
            'stage2': {},
            'stage3': {},
            'stage4': {}
        }
        self.results = {}
        
    def load_and_preprocess_data(self, data_path, augment=True):
        """Load dataset with augmentation"""
        print("Loading and preprocessing dataset...")
        
        images = []
        hybrid_features = []
        labels = []
        class_names = []
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset path {data_path} not found")
        
        subjects = sorted([d for d in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, d))])
        
        for idx, subject in enumerate(subjects):
            class_names.append(subject)
            subject_path = os.path.join(data_path, subject)
            
            for img_name in os.listdir(subject_path):
                img_path = os.path.join(subject_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Preprocess
                face = detect_and_align_face(img)
                features = extract_hybrid_features(face)
                
                images.append(face.reshape(128, 128, 1))
                hybrid_features.append(features)
                labels.append(idx)
                
                # Data augmentation
                if augment:
                    # Horizontal flip
                    face_flip = cv2.flip(face, 1)
                    features_flip = extract_hybrid_features(face_flip)
                    
                    images.append(face_flip.reshape(128, 128, 1))
                    hybrid_features.append(features_flip)
                    labels.append(idx)
        
        return (np.array(images), 
                np.array(hybrid_features), 
                np.array(labels), 
                class_names)
    
    def stage1_pretrain_cnn(self, X_img_train, X_feat_train, y_train, epochs=20):
        """
        Stage 1: Pretrain CNN branch alone on raw images
        """
        print("\n" + "="*60)
        print("STAGE 1: Pretraining CNN Branch")
        print("="*60)
        
        # Build CNN-only model
        cnn_model = build_cnn_only_model(
            img_shape=self.img_shape, 
            num_classes=self.num_classes
        )
        
        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                'weights/stage1_cnn_best.h5', 
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = cnn_model.fit(
            X_img_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['stage1'] = history.history
        self.cnn_model = cnn_model
        
        print(f"Stage 1 Complete - Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
        return cnn_model
    
    def stage2_pretrain_features(self, X_img_train, X_feat_train, y_train, epochs=10):
        """
        Stage 2: Pretrain feature branch alone on traditional descriptors
        """
        print("\n" + "="*60)
        print("STAGE 2: Pretraining Feature Branch")
        print("="*60)
        
        # Build feature-only model
        feat_model = build_feature_only_model(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes
        )
        
        feat_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                'weights/stage2_features_best.h5',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = feat_model.fit(
            X_feat_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['stage2'] = history.history
        self.feat_model = feat_model
        
        print(f"Stage 2 Complete - Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
        return feat_model
    
    def stage3_joint_finetuning(self, X_img_train, X_feat_train, y_train, 
                                 X_img_val, X_feat_val, y_val, epochs=50):
        """
        Stage 3: Joint fine-tuning of the full hybrid model
        """
        print("\n" + "="*60)
        print("STAGE 3: Joint Fine-Tuning Hybrid Model")
        print("="*60)
        
        # Build full hybrid model
        hybrid_model = build_hybrid_model(
            img_shape=self.img_shape,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes
        )
        
        # Custom loss function combining cross-entropy and center loss
        def hybrid_loss(y_true, y_pred):
            # Standard cross-entropy loss
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return ce_loss
        
        hybrid_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss={
                'output': 'sparse_categorical_crossentropy',
                'embedding': lambda y_true, y_pred: tf.zeros_like(y_pred)  # Placeholder
            },
            loss_weights={'output': 1.0, 'embedding': 0.0},
            metrics={'output': 'accuracy'}
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                'weights/best_hybrid_model.h5',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger('results/training_log.csv', append=True)
        ]
        
        # Train with dual outputs
        history = hybrid_model.fit(
            {'image_input': X_img_train, 'feature_input': X_feat_train},
            {'output': y_train, 'embedding': y_train},  # Dummy for embedding
            validation_data=(
                {'image_input': X_img_val, 'feature_input': X_feat_val},
                {'output': y_val, 'embedding': y_val}
            ),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['stage3'] = history.history
        self.hybrid_model = hybrid_model
        
        print(f"Stage 3 Complete - Best Val Accuracy: {max(history.history['val_output_accuracy']):.4f}")
        return hybrid_model
    
    def stage4_ensemble(self, models=None):
        """
        Stage 4: Ensemble multiple hybrid configurations (optional)
        """
        print("\n" + "="*60)
        print("STAGE 4: Ensemble Configuration")
        print("="*60)
        
        # For now, we use the best model from stage 3
        # Advanced: Load multiple models and ensemble
        print("Using best model from Stage 3 as final model")
        print("Advanced ensemble can be implemented with multiple model variants")
        
        return self.hybrid_model
    
    def train_complete(self, data_path, use_augmentation=True):
        """
        Execute complete 4-stage training pipeline
        """
        start_time = datetime.now()
        
        # Load data
        X_img, X_feat, y, class_names = self.load_and_preprocess_data(
            data_path, 
            augment=use_augmentation
        )
        
        if X_img is None:
            print("Failed to load data. Exiting.")
            return None
        
        # Split data
        X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = \
            train_test_split(X_img, X_feat, y, test_size=0.2, random_state=42, stratify=y)
        
        # Further split train into train/val for stage 3
        X_img_train, X_img_val, X_feat_train, X_feat_val, y_train, y_val = \
            train_test_split(X_img_train, X_feat_train, y_train, test_size=0.2, 
                           random_state=42, stratify=y_train)
        
        print(f"\nDataset Statistics:")
        print(f"  Train: {len(X_img_train)} samples")
        print(f"  Val:   {len(X_img_val)} samples")
        print(f"  Test:  {len(X_img_test)} samples")
        print(f"  Classes: {self.num_classes}")
        print(f"  Feature dim: {self.feature_dim}")
        
        # Execute training stages
        self.stage1_pretrain_cnn(X_img_train, X_feat_train, y_train, epochs=20)
        self.stage2_pretrain_features(X_img_train, X_feat_train, y_train, epochs=10)
        self.stage3_joint_finetuning(
            X_img_train, X_feat_train, y_train,
            X_img_val, X_feat_val, y_val,
            epochs=50
        )
        final_model = self.stage4_ensemble()
        
        # Comprehensive evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Note: Model now has dual outputs, need to handle this
        # For evaluation, we only use the 'output'
        eval_model = Model(
            inputs=final_model.input,
            outputs=final_model.get_layer('output').output
        )
        
        metrics = compute_comprehensive_metrics(
            eval_model,
            {'image_input': X_img_test, 'feature_input': X_feat_test},
            y_test,
            class_names=class_names
        )
        
        # Save results
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        self.results = {
            'metrics': metrics,
            'training_duration': training_duration,
            'timestamp': datetime.now().isoformat(),
            'dataset': data_path,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim
        }
        
        # Save to file
        results_path = f'results/training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs('results', exist_ok=True)
        
        # Convert numpy types for JSON serialization
        serializable_results = json.loads(json.dumps(self.results, default=lambda x: x.tolist()))
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        print(f"Total training time: {training_duration:.2f} seconds")
        
        # Save final model
        final_model.save('weights/final_hybrid_model.h5')
        print("Final model saved to weights/final_hybrid_model.h5")
        
        return final_model


def train_hybrid_system(data_path='data/raw/orl'):
    """
    Main training function with multi-stage pipeline
    """
    # Count classes
    if not os.path.exists(data_path):
        print(f"Dataset path {data_path} not found.")
        return
    
    subjects = [d for d in os.listdir(data_path) 
                if os.path.isdir(os.path.join(data_path, d))]
    num_classes = len(subjects)
    
    # Feature dimension: SIFT(128) + HOG(8100) + Gabor(24) = 8252
    feature_dim = 8252
    
    # Create trainer
    trainer = MultiStageTrainer(
        num_classes=num_classes,
        feature_dim=feature_dim,
        img_shape=(128, 128, 1)
    )
    
    # Execute complete training
    model = trainer.train_complete(data_path, use_augmentation=True)
    
    return model


if __name__ == "__main__":
    train_hybrid_system()
