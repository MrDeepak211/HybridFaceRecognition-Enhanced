import os
import cv2
import numpy as np

def generate_synthetic_data(base_path='data/raw/orl', num_subjects=5, images_per_subject=10):
    """
    Generates synthetic face-like data for testing the pipeline.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    print(f"Generating synthetic dataset with {num_subjects} subjects...")
    
    for i in range(1, num_subjects + 1):
        subject_dir = os.path.join(base_path, f's{i}')
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
            
        for j in range(1, images_per_subject + 1):
            # Create a random image with a "face-like" circle in the middle
            img = np.random.randint(0, 50, (112, 92), dtype=np.uint8)
            cv2.circle(img, (46, 56), 30, (200), -1) # Face
            cv2.circle(img, (35, 45), 5, (0), -1) # Left eye
            cv2.circle(img, (57, 45), 5, (0), -1) # Right eye
            cv2.ellipse(img, (46, 75), (15, 5), 0, 0, 180, (0), 2) # Mouth
            
            # Add some noise/variation
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            img_path = os.path.join(subject_dir, f'{j}.pgm')
            cv2.imwrite(img_path, img)
            
    print("Synthetic dataset generated successfully.")

if __name__ == "__main__":
    generate_synthetic_data()
