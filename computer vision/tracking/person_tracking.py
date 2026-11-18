
from ultralytics import YOLO


class PersonTracker:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracked_frame = results[0].plot()

        tracked_persons = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes

            for box, track_id in zip(boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)

                # Calculate person center coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                tracked_persons.append({
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "center_x": center_x,
                    "center_y": center_y
                })

        return tracked_frame, tracked_persons
