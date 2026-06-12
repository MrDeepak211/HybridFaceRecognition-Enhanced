"""
Main Training Script — Hybrid Face Recognition System
=======================================================
Usage:
  python main.py --train --dataset orl --method hybrid --optimizer adam --activation softmax
  python main.py --train --dataset sheffield --method sift --optimizer adamax --activation softmax
  python main.py --train-svm --dataset orl --method hybrid --kernel rbf
  python main.py --evaluate --dataset orl
"""

import argparse, os, cv2, numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint, TensorBoard)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import (extract_by_method, extract_hybrid_features,
                                      extract_research_fusion)
from src.models.hybrid_model import (build_hybrid_model, build_mobilenetv2_model)
from src.classifiers.svm_classifier import SVMClassifier


# ── Dataset Loaders ──────────────────────────────────────────
def load_orl(root='data/ORL', size=(128,128)):
    X_img, X_feat, y = [], [], []
    if not os.path.exists(root):
        print(f"⚠️  ORL dataset not found at '{root}'. Download from kaggle.")
        return None, None, None
    for sid in sorted(os.listdir(root)):
        spath = os.path.join(root, sid)
        if not os.path.isdir(spath): continue
        label = int(sid.replace('s','')) - 1
        for fn in sorted(os.listdir(spath)):
            fp = os.path.join(spath, fn)
            img = cv2.imread(fp)
            if img is None: continue
            face = detect_and_align_face(img, size)
            X_img.append(face.reshape(*size, 1))
            y.append(label)
    print(f"ORL loaded: {len(y)} images, {len(set(y))} subjects")
    return np.array(X_img), np.array(y)


def load_sheffield(root='data/Sheffield', size=(128,128)):
    X_img, y = [], []
    if not os.path.exists(root):
        print(f"⚠️  Sheffield dataset not found at '{root}'.")
        return None, None
    for sid in sorted(os.listdir(root)):
        spath = os.path.join(root, sid)
        if not os.path.isdir(spath): continue
        try: label = int(sid) - 1
        except: continue
        for fn in sorted(os.listdir(spath)):
            fp = os.path.join(spath, fn)
            img = cv2.imread(fp)
            if img is None: continue
            face = detect_and_align_face(img, size)
            X_img.append(face.reshape(*size, 1))
            y.append(label)
    print(f"Sheffield loaded: {len(y)} images, {len(set(y))} subjects")
    return np.array(X_img), np.array(y)


def extract_all_features(X_img, method='hybrid', fusion='research'):
    feats = []
    for img in tqdm(X_img, desc=f"Extracting [{method}]"):
        face = img.squeeze()
    if method == 'hybrid':
        f = extract_research_fusion(face) if fusion == 'research' else extract_hybrid_features(face)
    else:
        f = extract_by_method(face, method)
        feats.append(f)
    return np.array(feats)


def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.15),
    ])


# ── Train CNN ────────────────────────────────────────────────
def train_cnn(args):
    print(f"\n{'='*50}")
    print(f"Training: {args.model.upper()} | Dataset: {args.dataset.upper()}")
    print(f"Method: {args.method} | Optimizer: {args.optimizer} | Activation: {args.activation}")
    print('='*50)

    # Load data
    if args.dataset == 'orl':
        X_img, y = load_orl()
    else:
        X_img, y = load_sheffield()
    if X_img is None:
        print("Dataset not found. Exiting.")
        return

    num_classes = len(np.unique(y))

    # Extract features
    X_feat = extract_all_features(X_img, args.method, args.fusion)

    # Split
    Xi_tr, Xi_te, Xf_tr, Xf_te, y_tr, y_te = train_test_split(
        X_img, X_feat, y, test_size=0.2, stratify=y, random_state=42)

    # One-hot
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes)
    y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes)

    # Build model
    feat_dim = X_feat.shape[1]
    if args.model == 'mobilenetv2':
        model = build_mobilenetv2_model(img_shape=(96,96,3),
                                         feature_dim=feat_dim,
                                         num_classes=num_classes)
        # Convert grayscale → RGB for MobileNetV2
        Xi_tr = np.array([cv2.cvtColor((i.squeeze()*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                           for i in Xi_tr]).reshape(-1,96,96,3) / 255.0
        Xi_te = np.array([cv2.cvtColor((i.squeeze()*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                           for i in Xi_te]).reshape(-1,96,96,3) / 255.0
    else:
        model = build_hybrid_model(img_shape=(128,128,1),
                                    feature_dim=feat_dim,
                                    num_classes=num_classes)

    # Optimizer
    opt_map = {
        'adam':    tf.keras.optimizers.Adam(1e-3),
        'adamax':  tf.keras.optimizers.Adamax(1e-3),
        'rmsprop': tf.keras.optimizers.RMSprop(1e-3),
        'sgd':     tf.keras.optimizers.SGD(1e-2, momentum=0.9),
    }
    opt = opt_map.get(args.optimizer, tf.keras.optimizers.Adam(1e-3))

    # Loss based on activation
    loss = 'binary_crossentropy' if args.activation == 'sigmoid' else 'categorical_crossentropy'
    model.compile(optimizer=opt, loss={
        'output': loss, 'embedding': None
    } if 'embedding' in [l.name for l in model.layers] else loss,
        metrics=['accuracy'])

    # Callbacks
    os.makedirs('weights', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    tag = f"{args.model}_{args.dataset}_{args.method}_{args.optimizer}"
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f'weights/best_{tag}.h5', save_best_only=True, verbose=1),
        TensorBoard(log_dir=f'logs/{tag}'),
    ]

    # Augmentation
    aug = get_augmentation()

    print(f"\nModel parameters: {model.count_params():,}")
    print(f"Training samples:  {len(y_tr)}")
    print(f"Test samples:      {len(y_te)}")
    print(f"Feature dimension: {feat_dim}")
    print(f"Classes:           {num_classes}\n")

    history = model.fit(
        {'image_input': Xi_tr, 'feature_input': Xf_tr},
        y_tr_oh, validation_data=({'image_input': Xi_te, 'feature_input': Xf_te}, y_te_oh),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=1
    )

    # Evaluate
    results = model.evaluate(
        {'image_input': Xi_te, 'feature_input': Xf_te}, y_te_oh, verbose=0)
    print(f"\n✅ Test Accuracy: {results[1]*100:.2f}%")

    # Save final
    model.save(f'weights/best_{tag}_final.h5')
    print(f"Model saved → weights/best_{tag}_final.h5")
    return history


# ── Train SVM ────────────────────────────────────────────────
def train_svm(args):
    print(f"\nTraining SVM ({args.kernel}) on {args.dataset.upper()}")
    if args.dataset == 'orl':
        X_img, y = load_orl()
    else:
        X_img, y = load_sheffield()
    if X_img is None: return

    X_feat = extract_all_features(X_img, args.method, args.fusion)
    Xi_tr, Xi_te, Xf_tr, Xf_te, y_tr, y_te = train_test_split(
        X_img, X_feat, y, test_size=0.2, stratify=y, random_state=42)

    svm = SVMClassifier(kernel=args.kernel).build()
    print("Fitting SVM...")
    svm.train(Xf_tr, y_tr)
    result = svm.evaluate(Xf_te, y_te)
    print(f"\n✅ SVM ({args.kernel}) Accuracy: {result['accuracy_pct']}")
    print(result['report'])
    svm.save(f'weights/svm_{args.kernel}.pkl')
    print(f"SVM saved → weights/svm_{args.kernel}.pkl")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hybrid Face Recognition Training')
    p.add_argument('--train',     action='store_true', help='Train CNN/MobileNetV2')
    p.add_argument('--train-svm', action='store_true', help='Train SVM classifier')
    p.add_argument('--evaluate',  action='store_true', help='Evaluate model')
    p.add_argument('--dataset',   default='orl',       choices=['orl','sheffield'])
    p.add_argument('--method',    default='hybrid',    choices=['sift','hog','gabor','canny','hybrid'])
    p.add_argument('--fusion',    default='research',  choices=['research','full'])
    p.add_argument('--model',     default='custom',    choices=['custom','mobilenetv2'])
    p.add_argument('--optimizer', default='adam',      choices=['adam','adamax','rmsprop','sgd'])
    p.add_argument('--activation',default='softmax',   choices=['softmax','sigmoid'])
    p.add_argument('--kernel',    default='rbf',       choices=['rbf','linear','poly'])
    p.add_argument('--epochs',    default=100, type=int)
    p.add_argument('--batch-size',default=16,  type=int)
    args = p.parse_args()

    if args.train:     train_cnn(args)
    elif args.train_svm: train_svm(args)
    else: p.print_help()
