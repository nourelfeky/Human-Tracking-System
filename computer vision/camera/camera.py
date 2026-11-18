
import cv2


def start_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera started successfully!")
    print("Press 'q' to close the camera.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read frame")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        cv2.imshow("Human Tracking System - Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()

