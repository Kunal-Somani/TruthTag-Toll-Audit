import cv2
import os
from Centroid_Tracker import ClassicalTracker

def run_detector(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    tracker = ClassicalTracker() # Initialize the tracker
    
    TRIPWIRE_Y = 400 
    OFFSET = 10 
    saved_ids = set() # Keeps track of who we already photographed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        fg_mask = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.line(frame, (0, TRIPWIRE_Y), (frame.shape[1], TRIPWIRE_Y), (0, 0, 255), 2)

        # 1. Collect all bounding boxes first
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) > 5000: 
                x, y, w, h = cv2.boundingRect(contour)
                rects.append([x, y, w, h])

        # 2. Let the tracker assign IDs
        tracked_objects = tracker.update(rects)

        # 3. Process the tracked vehicles
        for obj in tracked_objects:
            x, y, w, h, obj_id = obj
            center_y = y + (h // 2)

            # Draw the box and the unique ID
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. Tripwire Logic + ID check
            if (TRIPWIRE_Y - OFFSET) < center_y < (TRIPWIRE_Y + OFFSET):
                if obj_id not in saved_ids:
                    cropped_img = frame[y:y+h, x:x+w]
                    if cropped_img.size > 0:
                        filename = os.path.join(output_dir, f"vehicle_ID{obj_id}.jpg")
                        cv2.imwrite(filename, cropped_img)
                        print(f"Saved: {filename}")
                        
                        saved_ids.add(obj_id) # Never crop this car again
                        cv2.line(frame, (0, TRIPWIRE_Y), (frame.shape[1], TRIPWIRE_Y), (0, 255, 0), 5)

        cv2.imshow('Motion Mask', fg_mask)
        cv2.imshow('Toll Camera Feed', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detector('../data/raw_video/test_video.mp4', '../data/extracted_crops')