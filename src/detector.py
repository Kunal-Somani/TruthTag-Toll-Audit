import cv2
import os

def run_detector(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    
    # 1. Background Subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    # 2. Virtual Tripwire Y-Coordinate (Adjust this based on your video)
    TRIPWIRE_Y = 400 
    OFFSET = 10 # Margin of error for line crossing
    
    car_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Apply Mask & Clean Shadows
        fg_mask = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY) # Removes gray shadows
        
        # 4. Morphological Cleaning (Fill holes in the blob)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # 5. Find Contours (The outlines of the blobs)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the Tripwire
        cv2.line(frame, (0, TRIPWIRE_Y), (frame.shape[1], TRIPWIRE_Y), (0, 0, 255), 2)

        for contour in contours:
            # Ignore tiny noise blobs
            if cv2.contourArea(contour) < 5000: 
                continue

            # Get Bounding Box
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + (h // 2)

            # Draw the box for visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 6. Tripwire Logic: Did the center of the car cross the line?
            if (TRIPWIRE_Y - OFFSET) < center_y < (TRIPWIRE_Y + OFFSET):
                # Crop and Save
                cropped_img = frame[y:y+h, x:x+w]
                if cropped_img.size > 0:
                    filename = os.path.join(output_dir, f"vehicle_{car_count}.jpg")
                    cv2.imwrite(filename, cropped_img)
                    print(f"Saved: {filename}")
                    car_count += 1
                    
                    # Flash the line green to show a capture
                    cv2.line(frame, (0, TRIPWIRE_Y), (frame.shape[1], TRIPWIRE_Y), (0, 255, 0), 5)

        # Show feeds
        cv2.imshow('Motion Mask', fg_mask)
        cv2.imshow('Toll Camera Feed', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Update this path to your actual video file
    run_detector('../data/raw_video/test_video.mp4', '../data/extracted_crops')