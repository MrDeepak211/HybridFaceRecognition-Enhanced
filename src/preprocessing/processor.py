"""
Preprocessing Module
=====================
Face Detection → Eye-Landmark Alignment → Rotation → Crop → Resize → Normalize
"""

import cv2
import numpy as np
import os


def detect_and_align_face(image, target_size=(128, 128)):
    """
    Full pipeline:
      1. Convert to grayscale
      2. Detect face (Haar Cascade)
      3. Detect eyes within face ROI
      4. Compute eye-centre angle → rotate for alignment
      5. Crop aligned face → resize → normalize [0,1]
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # ── Step 1: Detect face ───────────────────────────────
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # Fallback: centre-crop
        h, w = gray.shape
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        face_roi  = gray[top:top+side, left:left+side]
        resized   = cv2.resize(face_roi, target_size)
        return resized / 255.0

    # Largest face
    x, y, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_gray = gray[y:y+fh, x:x+fw]

    # ── Step 2: Eye-landmark detection inside face ROI ────
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)

    if len(eyes) >= 2:
        # Sort eyes left → right by x-centre
        eye_centres = sorted(
            [(ex + ew//2, ey + eh//2) for (ex, ey, ew, eh) in eyes],
            key=lambda c: c[0]
        )
        left_eye  = eye_centres[0]   # (cx, cy) relative to face_gray
        right_eye = eye_centres[1]

        # ── Step 3: Compute rotation angle ───────────────
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # ── Step 4: Rotate face image ─────────────────────
        face_h, face_w = face_gray.shape
        centre = (face_w // 2, face_h // 2)
        M = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
        aligned = cv2.warpAffine(face_gray, M, (face_w, face_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    else:
        # No eyes found — use raw face crop
        aligned = face_gray

    # ── Step 5: Resize & Normalize ────────────────────────
    resized    = cv2.resize(aligned, target_size)
    normalized = resized / 255.0
    return normalized


def preprocess_image(img_path, size=(128, 128)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return detect_and_align_face(img, size)
