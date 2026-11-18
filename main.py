
import cv2
import sys

sys.path.append("computer vision/tracking")

from person_tracking import PersonTracker


def main():
    tracker = PersonTracker()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    print("Person Tracking Started!")
    print("Press 'q' to exit.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)

        tracked_frame, tracked_persons = tracker.track(frame)

        if tracked_persons:
            for person in tracked_persons:
                print(
                    f"ID: {person['id']} | "
                    f"Center X: {person['center_x']} | "
                    f"Center Y: {person['center_y']}       "
                )
        else:
            print("No person detected                    ", end="\r")

        cv2.imshow("Human Tracking System", tracked_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
