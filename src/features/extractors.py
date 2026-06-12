"""
Feature Extraction Module
==========================
Methods : SIFT | HOG | Gabor | Canny
Fusion  : SIFT + HOG + Gabor          ← Research Fusion (Mam's paper focus)
Full    : SIFT + HOG + Gabor + Canny  ← Full Hybrid Fusion
"""

import cv2
import numpy as np
from skimage.feature import hog


# ── 1. SIFT (128-dim) ─────────────────────────────────────
def extract_sift_features(image, n_features=500):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=3,
                            contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    _, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(128)
    return np.mean(descriptors, axis=0)


# ── 2. HOG (gradient histogram) ───────────────────────────
def extract_hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False, channel_axis=None)


# ── 3. Gabor (24-dim filter bank) ─────────────────────────
def extract_gabor_features(image, frequencies=(0.1, 0.3, 0.5), thetas=(0, 45, 90, 135)):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    features = []
    for freq in frequencies:
        for theta in thetas:
            kernel = cv2.getGaborKernel((21, 21), sigma=4, theta=np.deg2rad(theta),
                                        lambd=1.0/freq, gamma=0.5, psi=0)
            filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
            features.extend([filtered.mean(), filtered.std()])
    return np.array(features)


# ── 4. Canny (65-dim edge density) ────────────────────────
def extract_canny_features(image, low_thresh=50, high_thresh=150, bins=64):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    edges = cv2.Canny(image, low_thresh, high_thresh)
    density = float(np.sum(edges > 0)) / edges.size
    spatial_hist = np.zeros(bins)
    flat_edges = edges.flatten()
    edge_positions = np.where(flat_edges > 0)[0]
    if len(edge_positions) > 0:
        bin_indices = np.clip((edge_positions * bins // len(flat_edges)).astype(int), 0, bins-1)
        for idx in bin_indices:
            spatial_hist[idx] += 1
        if spatial_hist.sum() > 0:
            spatial_hist /= spatial_hist.sum()
    return np.concatenate([[density], spatial_hist])


# ── Individual dispatcher ──────────────────────────────────
METHOD_MAP = {
    "sift":  extract_sift_features,
    "hog":   extract_hog_features,
    "gabor": extract_gabor_features,
    "canny": extract_canny_features,
}

def extract_by_method(image, method="sift"):
    method = method.lower().strip()
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method '{method}'. Choose: {list(METHOD_MAP.keys())}")
    return METHOD_MAP[method](image)


# ── Research Fusion: SIFT + HOG + Gabor (Mam's paper focus) ─
def extract_research_fusion(image):
    """SIFT + HOG + Gabor — matches Mam's proposed fusion."""
    return np.concatenate([
        extract_sift_features(image),   # 128
        extract_hog_features(image),    # variable
        extract_gabor_features(image),  # 24
    ])


# ── Full Hybrid: SIFT + HOG + Gabor + Canny ───────────────
def extract_hybrid_features(image):
    """All 4 methods combined — full hybrid fusion."""
    return np.concatenate([
        extract_sift_features(image),   # 128
        extract_hog_features(image),    # variable
        extract_gabor_features(image),  # 24
        extract_canny_features(image),  # 65
    ])
