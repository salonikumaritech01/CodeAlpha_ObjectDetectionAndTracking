import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("Traffic.mp4")

cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Tracking", 1200, 700)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Track objects
    results = model.track(
    frame,
    persist=True,
    conf=0.3,
    tracker="botsort.yaml"
)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Tracking", annotated_frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()