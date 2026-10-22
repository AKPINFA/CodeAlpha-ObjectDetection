import time

import cv2
from ultralytics import YOLO

CONFIDENCE = 0.5  # minimum confidence to keep a detection (0 to 1)

# Colors (B, G, R) used to draw the boxes, one per class
COLORS = [
    (86, 180, 233), (0, 158, 115), (240, 228, 66), (213, 94, 0),
    (204, 121, 167), (0, 114, 178), (230, 159, 0), (153, 153, 153),
]

# Pre-trained YOLO model: it is downloaded automatically the first time
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open the webcam.")

print("Detection started. Press 'q' to quit.")
previous_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Cannot read a frame from the webcam.")
        break

    # Run the model on the current frame
    results = model(frame, conf=CONFIDENCE, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # corners of the box
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = f"{model.names[class_id]} {confidence:.0%}"
        color = COLORS[class_id % len(COLORS)]

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with a colored background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Frames per second
    now = time.time()
    fps = 1 / (now - previous_time)
    previous_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Object detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()