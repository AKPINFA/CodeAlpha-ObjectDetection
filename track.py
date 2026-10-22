import time

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

CONFIDENCE = 0.5  # minimum confidence to keep a detection (0 to 1)

# Colors (B, G, R) used to draw the boxes, one per track ID
COLORS = [
    (86, 180, 233), (0, 158, 115), (240, 228, 66), (213, 94, 0),
    (204, 121, 167), (0, 114, 178), (230, 159, 0), (153, 153, 153),
]

# 1) Detector: pre-trained YOLO model
model = YOLO("yolo11n.pt")

# 2) Tracker: Deep SORT
#    max_age: how many frames a lost object is remembered
#    n_init : how many frames before a new object gets a confirmed ID
tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet", embedder_gpu=False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open the webcam.")

print("Tracking started. Press 'q' to quit.")
previous_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Cannot read a frame from the webcam.")
        break

    # Step A: detect objects in the frame
    results = model(frame, conf=CONFIDENCE, verbose=False)[0]

    # Step B: convert detections to the format expected by Deep SORT:
    #         ([left, top, width, height], confidence, class)
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), int(box.cls[0])))

    # Step C: update the tracker -> every object gets an ID that stays the same
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        # Ignore tracks not yet confirmed or not seen in the current frame
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        class_name = model.names[track.get_det_class()]
        color = COLORS[int(track_id) % len(COLORS)]
        label = f"{class_name} #{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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

    cv2.imshow("Object detection and tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()