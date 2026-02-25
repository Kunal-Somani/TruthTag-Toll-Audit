import cv2
import os
import shutil

def rapid_labeler(input_folder, output_base):
    # Create the sorted folders
    classes = {'c': 'car', 't': 'truck', 'v': 'van', 'x': 'trash'}
    for cls in classes.values():
        os.makedirs(os.path.join(output_base, cls), exist_ok=True)

    images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print("Controls: [C]ar | [T]ruck | [V]an | [X]Trash/Ignore | [Q]uit")

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        cv2.imshow('Labeler - Press key to sort', img)
        key = cv2.waitKey(0) & 0xFF 

        # Map the key press to the folder
        char_key = chr(key).lower()
        
        if char_key == 'q':
            print("Quitting...")
            break
        elif char_key in classes:
            target_folder = os.path.join(output_base, classes[char_key])
            shutil.move(img_path, os.path.join(target_folder, img_name))
            print(f"Moved to {classes[char_key]}")
        else:
            print("Invalid key. Skipping...")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    rapid_labeler('../data/unlabelled', '../data/labeled_dataset')