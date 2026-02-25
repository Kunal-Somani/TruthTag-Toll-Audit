import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

def get_hog_fingerprint(image_path, visualize=True):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    # 2. Resize to a strict standard (Crucial for SVM later)
    # HOG requires a 1:2 aspect ratio. 64x128 is the industry standard.
    resized_img = cv2.resize(img, (64, 128))
    
    # 3. Convert to Grayscale (HOG only cares about shapes, not color)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    # 4. Extract HOG Features
    features, hog_image = hog(
        gray_img, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        visualize=True, 
        block_norm='L2-Hys' # Fixes lighting/shadow issues
    )
    
    if visualize:
        # Show the original resized image and its HOG fingerprint
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Resized')

        ax2.axis('off')
        ax2.imshow(hog_image, cmap=plt.cm.gray)
        ax2.set_title('HOG Fingerprint (Gradients)')
        plt.show()

    return features

if __name__ == "__main__":
    # We will test it on a single image first
    test_image = 'dataset/train/cars/test_car.jpg' 
    features = get_hog_fingerprint(test_image, visualize=True)