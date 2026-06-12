"""
Dataset Management and Preprocessing Tools
Supports: ORL, Extended Yale B, AR, FERET, LFW, Sheffield
"""

import os
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil


class DatasetManager:
    """
    Manages dataset downloading, preprocessing, and splitting
    """
    
    DATASETS = {
        'orl': {
            'url': 'https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip',
            'description': 'ORL (AT&T) Face Database - 400 images, 40 subjects',
            'format': 'pgm',
            'subjects': 40,
            'images_per_subject': 10
        },
        'yale_b': {
            'url': None,  # Manual download required
            'description': 'Extended Yale B - 2414 images, 38 subjects',
            'format': 'pgm',
            'subjects': 38
        },
        'lfw': {
            'url': 'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
            'description': 'Labeled Faces in the Wild - 13,233 images, 5,749 subjects',
            'format': 'jpg',
            'subjects': 5749
        }
    }
    
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'raw')
        self.processed_dir = os.path.join(base_dir, 'processed')
        self.splits_dir = os.path.join(base_dir, 'splits')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)
    
    def download_dataset(self, dataset_name):
        """
        Download dataset (if URL available)
        """
        if dataset_name not in self.DATASETS:
            print(f"Dataset {dataset_name} not supported")
            return False
        
        dataset_info = self.DATASETS[dataset_name]
        
        if dataset_info['url'] is None:
            print(f"\n{dataset_info['description']}")
            print(f"Manual download required. Please visit the dataset website.")
            print(f"Place the downloaded files in: {self.raw_dir}/{dataset_name}")
            return False
        
        print(f"Downloading {dataset_name}...")
        print(f"Description: {dataset_info['description']}")
        
        try:
            save_path = os.path.join(self.raw_dir, f"{dataset_name}.zip")
            
            with tqdm(unit='B', unit_scale=True, miniters=1) as t:
                def progress_hook(blocknum, blocksize, totalsize):
                    t.total = totalsize
                    t.update(blocknum * blocksize - t.n)
                
                urllib.request.urlretrieve(
                    dataset_info['url'],
                    save_path,
                    reporthook=progress_hook
                )
            
            print(f"\nDownloaded to {save_path}")
            print("Extracting...")
            
            # Extract
            if save_path.endswith('.zip'):
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_dir)
            elif save_path.endswith('.tar.gz') or save_path.endswith('.tgz'):
                with tarfile.open(save_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(self.raw_dir)
            
            print(f"Extraction complete to {self.raw_dir}/{dataset_name}")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def preprocess_dataset(self, dataset_name, output_size=(128, 128)):
        """
        Preprocess dataset: detect faces, align, resize, normalize
        """
        raw_path = os.path.join(self.raw_dir, dataset_name)
        processed_path = os.path.join(self.processed_dir, dataset_name)
        
        if not os.path.exists(raw_path):
            print(f"Raw dataset not found at {raw_path}")
            return False
        
        print(f"\nPreprocessing {dataset_name}...")
        os.makedirs(processed_path, exist_ok=True)
        
        from src.preprocessing.processor import detect_and_align_face
        
        subjects = sorted([d for d in os.listdir(raw_path) 
                          if os.path.isdir(os.path.join(raw_path, d))])
        
        total_processed = 0
        total_failed = 0
        
        for subject in tqdm(subjects, desc="Subjects"):
            subject_path = os.path.join(raw_path, subject)
            processed_subject_path = os.path.join(processed_path, subject)
            os.makedirs(processed_subject_path, exist_ok=True)
            
            images = [f for f in os.listdir(subject_path) 
                     if f.endswith(('.pgm', '.jpg', '.jpeg', '.png'))]
            
            for img_file in images:
                img_path = os.path.join(subject_path, img_file)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        total_failed += 1
                        continue
                    
                    # Detect and align face
                    face = detect_and_align_face(img, target_size=output_size)
                    
                    # Save processed image
                    output_file = os.path.join(processed_subject_path, img_file.replace('.pgm', '.png'))
                    cv2.imwrite(output_file, (face * 255).astype(np.uint8))
                    
                    total_processed += 1
                    
                except Exception as e:
                    total_failed += 1
                    continue
        
        print(f"\nPreprocessing complete:")
        print(f"  Success: {total_processed}")
        print(f"  Failed:  {total_failed}")
        
        return total_processed > 0
    
    def create_splits(self, dataset_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Create train/val/test splits
        """
        processed_path = os.path.join(self.processed_dir, dataset_name)
        splits_path = os.path.join(self.splits_dir, dataset_name)
        
        if not os.path.exists(processed_path):
            print(f"Processed dataset not found at {processed_path}")
            return False
        
        print(f"\nCreating splits for {dataset_name}...")
        print(f"  Train: {train_ratio*100:.0f}%")
        print(f"  Val:   {val_ratio*100:.0f}%")
        print(f"  Test:  {test_ratio*100:.0f}%")
        
        os.makedirs(splits_path, exist_ok=True)
        
        subjects = sorted([d for d in os.listdir(processed_path) 
                          if os.path.isdir(os.path.join(processed_path, d))])
        
        train_data = []
        val_data = []
        test_data = []
        
        for subject in subjects:
            subject_path = os.path.join(processed_path, subject)
            images = [os.path.join(subject_path, f) 
                     for f in os.listdir(subject_path)
                     if f.endswith('.png')]
            
            # Split images for this subject
            train_imgs, temp_imgs = train_test_split(
                images, train_size=train_ratio, random_state=42
            )
            
            val_size = val_ratio / (val_ratio + test_ratio)
            val_imgs, test_imgs = train_test_split(
                temp_imgs, train_size=val_size, random_state=42
            )
            
            train_data.extend(train_imgs)
            val_data.extend(val_imgs)
            test_data.extend(test_imgs)
        
        # Save splits
        split_files = {
            'train.txt': train_data,
            'val.txt': val_data,
            'test.txt': test_data
        }
        
        for filename, data in split_files.items():
            filepath = os.path.join(splits_path, filename)
            with open(filepath, 'w') as f:
                for img_path in data:
                    f.write(f"{img_path}\n")
        
        print(f"\nSplits created:")
        print(f"  Train: {len(train_data)} images")
        print(f"  Val:   {len(val_data)} images")
        print(f"  Test:  {len(test_data)} images")
        
        return True
    
    def load_split(self, dataset_name, split='train'):
        """
        Load images from a specific split
        """
        splits_path = os.path.join(self.splits_dir, dataset_name, f"{split}.txt")
        
        if not os.path.exists(splits_path):
            print(f"Split file not found: {splits_path}")
            return None, None
        
        images = []
        labels = []
        
        with open(splits_path, 'r') as f:
            for line in f:
                img_path = line.strip()
                if not img_path:
                    continue
                
                img = cv2.imread(img_path)
                if img is not None:
                    # Extract label from path
                    subject_name = Path(img_path).parent.name
                    images.append(img)
                    labels.append(subject_name)
        
        return np.array(images), np.array(labels)
    
    def get_dataset_stats(self, dataset_name):
        """
        Get dataset statistics
        """
        processed_path = os.path.join(self.processed_dir, dataset_name)
        
        if not os.path.exists(processed_path):
            return None
        
        subjects = [d for d in os.listdir(processed_path) 
                   if os.path.isdir(os.path.join(processed_path, d))]
        
        total_images = 0
        images_per_subject = {}
        
        for subject in subjects:
            subject_path = os.path.join(processed_path, subject)
            count = len([f for f in os.listdir(subject_path) 
                        if f.endswith('.png')])
            images_per_subject[subject] = count
            total_images += count
        
        return {
            'dataset': dataset_name,
            'num_subjects': len(subjects),
            'total_images': total_images,
            'images_per_subject': images_per_subject,
            'avg_images_per_subject': total_images / len(subjects) if subjects else 0
        }


class DataAugmenter:
    """
    Advanced data augmentation for face recognition
    """
    
    @staticmethod
    def augment_image(image, augmentations=None):
        """
        Apply augmentations to a single image
        """
        if augmentations is None:
            augmentations = {
                'horizontal_flip': True,
                'rotation': 15,
                'brightness': 0.2,
                'translation': 0.1
            }
        
        augmented_images = [image]  # Original image
        
        # Horizontal flip
        if augmentations.get('horizontal_flip', False):
            flipped = cv2.flip(image, 1)
            augmented_images.append(flipped)
        
        # Rotation
        rotation_angle = augmentations.get('rotation', 0)
        if rotation_angle > 0:
            for angle in [-rotation_angle, rotation_angle]:
                rotated = DataAugmenter.rotate_image(image, angle)
                augmented_images.append(rotated)
        
        # Brightness
        brightness_factor = augmentations.get('brightness', 0)
        if brightness_factor > 0:
            for factor in [-brightness_factor, brightness_factor]:
                brightened = DataAugmenter.adjust_brightness(image, factor)
                augmented_images.append(brightened)
        
        return augmented_images
    
    @staticmethod
    def rotate_image(image, angle):
        """Rotate image by angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    
    @staticmethod
    def adjust_brightness(image, factor):
        """Adjust image brightness"""
        if len(image.shape) == 3:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = np.clip(v + factor * 255, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # Grayscale
            adjusted = np.clip(image + factor * 255, 0, 255).astype(np.uint8)
            return adjusted
    
    @staticmethod
    def augment_dataset(processed_path, output_path, augmentations=None):
        """
        Augment entire dataset
        """
        os.makedirs(output_path, exist_ok=True)
        
        subjects = [d for d in os.listdir(processed_path) 
                   if os.path.isdir(os.path.join(processed_path, d))]
        
        total_original = 0
        total_augmented = 0
        
        for subject in tqdm(subjects, desc="Augmenting"):
            subject_path = os.path.join(processed_path, subject)
            output_subject_path = os.path.join(output_path, subject)
            os.makedirs(output_subject_path, exist_ok=True)
            
            images = [f for f in os.listdir(subject_path) 
                     if f.endswith('.png')]
            
            for img_file in images:
                img_path = os.path.join(subject_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                total_original += 1
                
                # Apply augmentations
                augmented = DataAugmenter.augment_image(img, augmentations)
                
                for i, aug_img in enumerate(augmented):
                    if i == 0:
                        # Original
                        output_file = os.path.join(output_subject_path, img_file)
                    else:
                        # Augmented
                        base_name = img_file.replace('.png', '')
                        output_file = os.path.join(output_subject_path, f"{base_name}_aug{i}.png")
                    
                    cv2.imwrite(output_file, aug_img)
                    total_augmented += 1
        
        print(f"\nAugmentation complete:")
        print(f"  Original images: {total_original}")
        print(f"  Total images:    {total_augmented}")
        print(f"  Augmentation factor: {total_augmented/total_original:.1f}x")


def setup_complete_dataset(dataset_name='orl'):
    """
    Complete pipeline: download -> preprocess -> split -> augment
    """
    manager = DatasetManager()
    
    print(f"\n{'='*60}")
    print(f"SETTING UP {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Step 1: Download
    print("\n[Step 1/4] Downloading dataset...")
    manager.download_dataset(dataset_name)
    
    # Step 2: Preprocess
    print("\n[Step 2/4] Preprocessing images...")
    manager.preprocess_dataset(dataset_name)
    
    # Step 3: Create splits
    print("\n[Step 3/4] Creating train/val/test splits...")
    manager.create_splits(dataset_name)
    
    # Step 4: Augment (optional)
    print("\n[Step 4/4] Data augmentation (optional)...")
    processed_path = os.path.join(manager.processed_dir, dataset_name)
    augmented_path = os.path.join(manager.processed_dir, f"{dataset_name}_augmented")
    
    if os.path.exists(processed_path):
        DataAugmenter.augment_dataset(processed_path, augmented_path)
        manager.create_splits(f"{dataset_name}_augmented")
    
    # Show stats
    stats = manager.get_dataset_stats(dataset_name)
    if stats:
        print(f"\n{'='*60}")
        print(f"DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Subjects: {stats['num_subjects']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Avg images/subject: {stats['avg_images_per_subject']:.1f}")
    
    print(f"\nDataset setup complete!")
    return manager


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Management Tool')
    parser.add_argument('--dataset', type=str, default='orl', 
                       help='Dataset name (orl, yale_b, lfw)')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess images')
    parser.add_argument('--split', action='store_true', help='Create splits')
    parser.add_argument('--augment', action='store_true', help='Augment dataset')
    parser.add_argument('--setup-all', action='store_true', help='Complete setup pipeline')
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    if args.setup_all:
        setup_complete_dataset(args.dataset)
    else:
        if args.download:
            manager.download_dataset(args.dataset)
        if args.preprocess:
            manager.preprocess_dataset(args.dataset)
        if args.split:
            manager.create_splits(args.dataset)
