import math

class ClassicalTracker:
    def __init__(self):
        # Stores center positions of objects: {id: (x, y)}
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (2 * x + w) // 2
            cy = (2 * y + h) // 2

            # Check if this object was detected in the previous frame
            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                # Euclidean distance
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 50:  # 50 pixels is the tracking threshold
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True
                    break

            # If it's a new object, assign a new ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean up lost objects (dictionary comprehension)
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids