import cv2
import os

def convert_samples():
    source_dir = 'data/raw/orl/s1'
    target_dir = 'static/samples'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for i in range(1, 4):
        img_path = os.path.join(source_dir, f'{i}.pgm')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(target_dir, f'sample_person1_{i}.png'), img)
            print(f"Converted {img_path} to static/samples/sample_person1_{i}.png")

if __name__ == "__main__":
    convert_samples()
