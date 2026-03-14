
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os



#  CLAHE-based Histogram Equalization
def apply_clahe(image):
   
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)



#  Gaussian Smoothing (Noise Removal)
def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)


#  Gamma Correction (Illumination Normalization)
def apply_gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(image, table)


#  Laplacian Edge Enhancement (Sharpening)
def apply_edge_enhancement(image):
    
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    enhanced = cv2.addWeighted(image, 1.0, laplacian, 0.3, 0)
    return enhanced


#  Full Preprocessing Pipeline
def preprocess_image(image, target_size=(48, 48)):
   
    # Resize
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian noise removal
    image = apply_gaussian_blur(image)

    # CLAHE histogram equalization
    image = apply_clahe(image)

    # Gamma correction
    image = apply_gamma_correction(image, gamma=1.2)

    # Edge sharpening
    image = apply_edge_enhancement(image)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image



#  Load FER2013 from CSV
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def load_fer2013(csv_path):
    
    print(f"[INFO] Loading FER2013 from: {csv_path}")
    df = pd.read_csv(csv_path)

    images, labels = [], []

    for idx, row in df.iterrows():
        # Parse pixel string → 48x48 uint8 array
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        processed = preprocess_image(pixels)
        images.append(processed)
        labels.append(int(row['emotion']))

        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(df)} images...")

    X = np.array(images)
    y = np.array(labels)

    print(f"[INFO] Dataset shape: {X.shape}, Labels: {y.shape}")
    return X, y


def split_dataset(X, y, val_size=0.1, test_size=0.1, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=ratio, random_state=random_state, stratify=y_temp
    )
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Quick test with a synthetic image
    dummy = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    result = preprocess_image(dummy)
    print(f"[TEST] Preprocessed image shape: {result.shape}, min: {result.min():.3f}, max: {result.max():.3f}")
    print("[PASS] Preprocessing pipeline works correctly.")
