import os
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Import your custom Forensic Expert
from feature_extractor import get_hog_fingerprint

def load_and_extract(dataset_path, class_mapping, max_images=1000):
    X = [] # Holds the math features
    y = [] # Holds the labels
    
    for folder_name, label in class_mapping.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            continue
            
        print(f"Extracting {label} (from folder {folder_name})...")
        
        # Grab a limited number of images so your laptop doesn't freeze
        image_files = os.listdir(folder_path)[:max_images]
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            # The Handoff: Turn the image into numbers
            features = get_hog_fingerprint(img_path, visualize=False)
            
            if features is not None:
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

def train_model(train_dir, test_dir, model_save_path):
    print("--- STARTING HOG+SVM PIPELINE ---")
    
    # The grouping rule! Notice 'd' is ignored.
    class_mapping = {
        'f1': 'Car',
        'f2': 'LCV',
        'f3': 'Truck',
        'f4': 'Multi-Axle',
        'f5': 'Multi-Axle',
        'f6': 'Multi-Axle'
    }

    print("\n1. Loading TRAINING Data (The Study Material)...")
    # Taking up to 1500 images per class for training
    X_train, y_train = load_and_extract(train_dir, class_mapping, max_images=1500)

    print("\n2. Loading TESTING Data (The Pop Quiz)...")
    # Taking up to 400 images per class for testing
    X_test, y_test = load_and_extract(test_dir, class_mapping, max_images=400)

    print(f"\nTraining on {len(X_train)} images, Testing on {len(X_test)} images.")

    print("\n3. Training the Blind Judge (Linear SVM)...")
    start_time = time.time()
    
    # The actual AI Model
    svm_model = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    svm_model.fit(X_train, y_train)
    
    print(f"Training finished in {round(time.time() - start_time, 2)} seconds.")

    print("\n4. Evaluating Model Accuracy...")
    predictions = svm_model.predict(X_test)
    
    # Print the detailed report card
    print(classification_report(y_test, predictions))
    
    acc = accuracy_score(y_test, predictions)
    print(f"OVERALL ACCURACY: {round(acc * 100, 2)}%")

    # Save the brain to a file
    joblib.dump(svm_model, model_save_path)
    print(f"\nModel saved successfully to {model_save_path}")

if __name__ == "__main__":
    TRAIN_DIR = '../dataset/cropped_data'
    TEST_DIR = '../dataset/cropped_test_data'
    MODEL_SAVE_PATH = 'svm_vehicle_model.pkl'
    
    train_model(TRAIN_DIR, TEST_DIR, MODEL_SAVE_PATH)