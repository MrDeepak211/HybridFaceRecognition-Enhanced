"""
Model Architecture Module
==========================
Hybrid Face Recognition System
Architectures:
  1. build_hybrid_model      — Custom CNN + Feature MLP (Mam's paper)
  2. build_mobilenetv2_model — MobileNetV2 Transfer Learning (proposed)
  3. build_cnn_only_model    — Baseline CNN
  4. build_feature_only_model— Baseline MLP
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ─────────────────────────────────────────────────────────────
# 1. Custom Hybrid CNN (Mam's Architecture — Enhanced)
# ─────────────────────────────────────────────────────────────
def build_hybrid_model(img_shape=(128, 128, 1), feature_dim=8317, num_classes=40):
    img_input = tf.keras.Input(shape=img_shape, name='image_input')

    # Conv Block 1 — 32 filters
    x = layers.Conv2D(32, (3,3), padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.2)(x)

    # Conv Block 2 — 64 filters
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.3)(x)

    # Conv Block 3 — 128 filters
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.3)(x)

    # Conv Block 4 — 256 filters
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Feature Branch (MLP)
    feat_input = tf.keras.Input(shape=(feature_dim,), name='feature_input')
    y = layers.Dense(1024, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # Fusion
    merged = layers.Concatenate()([x, y])
    f = layers.Dense(512, activation='relu')(merged)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.5)(f)
    f_res = layers.Dense(512, activation='relu')(merged)
    f = layers.Add()([f, f_res])
    f = layers.Dense(256, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.4)(f)
    f = layers.Dense(128, activation='relu')(f)
    embedding = layers.Dense(64, activation='relu', name='embedding')(f)
    output = layers.Dense(num_classes, activation='softmax', name='output')(embedding)

    return Model(inputs=[img_input, feat_input], outputs=[output, embedding])


# ─────────────────────────────────────────────────────────────
# 2. MobileNetV2 Transfer Learning (Proposed Enhancement)
# ─────────────────────────────────────────────────────────────
def build_mobilenetv2_model(img_shape=(96, 96, 3), feature_dim=8317, num_classes=40, fine_tune_at=100):
    """
    MobileNetV2 pretrained on ImageNet, fine-tuned on ORL/Sheffield.
    Dual-branch: MobileNetV2 image branch + feature MLP branch.
    """
    # RGB input for MobileNetV2
    img_input = tf.keras.Input(shape=img_shape, name='image_input')
    base = tf.keras.applications.MobileNetV2(
        input_shape=img_shape, include_top=False, weights='imagenet'
    )
    # Freeze base layers, unfreeze from fine_tune_at onward
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    x = base(img_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Feature MLP branch
    feat_input = tf.keras.Input(shape=(feature_dim,), name='feature_input')
    y = layers.Dense(512, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # Fusion
    merged = layers.Concatenate()([x, y])
    f = layers.Dense(256, activation='relu')(merged)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.4)(f)
    embedding = layers.Dense(64, activation='relu', name='embedding')(f)
    output = layers.Dense(num_classes, activation='softmax', name='output')(embedding)

    return Model(inputs=[img_input, feat_input], outputs=[output, embedding])


# ─────────────────────────────────────────────────────────────
# 3. Baseline CNN Only
# ─────────────────────────────────────────────────────────────
def build_cnn_only_model(img_shape=(128, 128, 1), num_classes=40):
    img_input = tf.keras.Input(shape=img_shape, name='image_input')
    x = layers.Conv2D(32, (3,3), padding='same')(img_input)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x); x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=img_input, outputs=output)


# ─────────────────────────────────────────────────────────────
# 4. Baseline Feature-Only MLP
# ─────────────────────────────────────────────────────────────
def build_feature_only_model(feature_dim=8317, num_classes=40):
    feat_input = tf.keras.Input(shape=(feature_dim,), name='feature_input')
    y = layers.Dense(1024, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y); y = layers.Dropout(0.4)(y)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.BatchNormalization()(y); y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu')(y); y = layers.Dropout(0.3)(y)
    output = layers.Dense(num_classes, activation='softmax')(y)
    return Model(inputs=feat_input, outputs=output)
