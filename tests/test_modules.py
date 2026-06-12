import unittest
import numpy as np
import cv2
import os
from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import extract_sift_features, extract_hog_features, extract_gabor_features

class TestHybridSystem(unittest.TestCase):
    def setUp(self):
        # Create a dummy gray image
        self.dummy_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        # Add a white circle to simulate a face for detection
        cv2.circle(self.dummy_img, (64, 64), 40, (255), -1)

    def test_preprocessing(self):
        """Test if preprocessing returns correctly shaped and normalized image."""
        processed = detect_and_align_face(self.dummy_img, target_size=(128, 128))
        self.assertEqual(processed.shape, (128, 128))
        self.assertTrue(0 <= processed.min() <= processed.max() <= 1.0)

    def test_sift_extraction(self):
        """Test SIFT feature extraction returns 128-dim vector."""
        features = extract_sift_features(self.dummy_img)
        self.assertEqual(features.shape, (128,))

    def test_hog_extraction(self):
        """Test HOG feature extraction returns a vector."""
        # Normalize dummy img to [0,1] for HOG if needed, 
        # but our extractor handles both
        features = extract_hog_features(self.dummy_img / 255.0)
        self.assertTrue(len(features) > 0)
        self.assertIsInstance(features, np.ndarray)

    def test_gabor_extraction(self):
        """Test Gabor feature extraction returns expected number of features."""
        # 3 frequencies * 4 thetas * 2 (mean & std) = 24 features
        features = extract_gabor_features(self.dummy_img)
        self.assertEqual(features.shape, (24,))

if __name__ == '__main__':
    unittest.main()
