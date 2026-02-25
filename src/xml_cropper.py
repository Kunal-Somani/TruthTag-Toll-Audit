import os
import cv2
import xml.etree.ElementTree as ET

def process_dataset(source_folder, output_folder):
    # Ensure the output base folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all XML files in the directory
    xml_files = [f for f in os.listdir(source_folder) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} XML files. Starting extraction...")

    count = 0
    for xml_file in xml_files:
        # 1. Parse the XML
        tree = ET.parse(os.path.join(source_folder, xml_file))
        root = tree.getroot()

        # 2. Get the matching JPG file name
        img_name = root.find('filename').text
        img_path = os.path.join(source_folder, img_name)
        
        # Fallback just in case the filename in XML doesn't match the actual file exactly
        if not os.path.exists(img_path):
            img_path = os.path.join(source_folder, xml_file.replace('.xml', '.jpg'))

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 3. Loop through all vehicles (objects) in this image
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower()
            
            # Create a folder for this specific class if it doesn't exist
            class_folder = os.path.join(output_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # 4. Get Bounding Box coordinates
            xmlbox = obj.find('bndbox')
            xmin = int(float(xmlbox.find('xmin').text))
            ymin = int(float(xmlbox.find('ymin').text))
            xmax = int(float(xmlbox.find('xmax').text))
            ymax = int(float(xmlbox.find('ymax').text))

            # Prevent out-of-bounds errors
            ymin, ymax = max(0, ymin), min(img.shape[0], ymax)
            xmin, xmax = max(0, xmin), min(img.shape[1], xmax)

            # 5. Crop the vehicle
            cropped_img = img[ymin:ymax, xmin:xmax]
            
            # Skip invalid crops (e.g., width or height is 0)
            if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                continue

            # 6. Resize to HOG Standard (64x128)
            resized_crop = cv2.resize(cropped_img, (64, 128))

            # 7. Save the perfectly cropped image
            save_path = os.path.join(class_folder, f"{count}.jpg")
            cv2.imwrite(save_path, resized_crop)
            count += 1

            if count % 1000 == 0:
                print(f"Processed {count} vehicles...")

    print(f"Done! Successfully extracted {count} cropped vehicles into '{output_folder}'.")

if __name__ == "__main__":
    # Point this to where you downloaded the dataset
    SOURCE_DIR = 'dataset/train'  
    # Where you want the perfectly cropped, resized images to go
    OUTPUT_DIR = 'dataset/cropped_data' 
    
    process_dataset(SOURCE_DIR, OUTPUT_DIR)