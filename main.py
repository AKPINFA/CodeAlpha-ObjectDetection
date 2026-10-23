import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Colors (B, G, R) used to draw the boxes, one per track ID
COLORS = [
    (86, 180, 233), (0, 158, 115), (240, 228, 66), (213, 94, 0),
    (204, 121, 167), (0, 114, 178), (230, 159, 0), (153, 153, 153),
]
TRAIL_LENGTH = 40  # number of past positions drawn behind each object


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object detection and tracking (YOLO + Deep SORT)."
    )
    parser.add_argument("--source", default="0",
                        help="webcam index (0, 1, ...) or path to a video file (default: 0)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO model weights (default: yolo11n.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="minimum detection confidence between 0 and 1 (default: 0.5)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="only track these classes, e.g. --classes person car")
    parser.add_argument("--save", action="store_true",
                        help="save the annotated video to output/result.mp4")
    return parser.parse_args()


def draw_label(frame, text, x, y, color):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_panel(frame, lines):
    """Semi-transparent information panel in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (270, 16 + 26 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (18, 34 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    args = parse_args()

    # ----- Detector -----
    model = YOLO(args.model)
    class_ids = None
    if args.classes:
        name_to_id = {name: i for i, name in model.names.items()}
        unknown = [c for c in args.classes if c not in name_to_id]
        if unknown:
            raise SystemExit(
                f"Unknown class(es): {', '.join(unknown)}. "
                "Examples: person, car, bottle, cell phone, chair"
            )
        class_ids = [name_to_id[c] for c in args.classes]

    # ----- Tracker -----
    tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet", embedder_gpu=False)

    # ----- Video source -----
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open the video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        Path("output").mkdir(exist_ok=True)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps < 1 or source_fps > 120:
            source_fps = 20
        writer = cv2.VideoWriter("output/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                 source_fps, (width, height))

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    ids_per_class = defaultdict(set)  # class name -> set of track IDs ever seen
    fps = 0.0
    frames = 0
    started = time.time()
    previous_time = started

    print("Started. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of the video file or camera error
            frames += 1

            # Detect
            results = model(frame, conf=args.conf, classes=class_ids, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1],
                                   float(box.conf[0]), int(box.cls[0])))

            # Track
            tracks = tracker.update_tracks(detections, frame=frame)

            visible = 0
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                visible += 1

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                class_name = model.names[track.get_det_class()]
                color = COLORS[int(track_id) % len(COLORS)]
                ids_per_class[class_name].add(track_id)

                # Trail: line through the last positions of the object
                points = trails[track_id]
                points.append(((x1 + x2) // 2, (y1 + y2) // 2))
                for i in range(1, len(points)):
                    thickness = max(1, int(4 * i / len(points)))
                    cv2.line(frame, points[i - 1], points[i], color, thickness)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, f"{class_name} #{track_id}", x1, y1, color)

            # Smoothed FPS
            now = time.time()
            current_fps = 1 / max(now - previous_time, 1e-6)
            fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
            previous_time = now

            total_unique = sum(len(ids) for ids in ids_per_class.values())
            lines = [f"FPS: {fps:.1f}", f"In frame: {visible}", f"Unique objects: {total_unique}"]
            for name, ids in sorted(ids_per_class.items(), key=lambda kv: -len(kv[1]))[:4]:
                lines.append(f"  {name}: {len(ids)}")
            draw_panel(frame, lines)

            if writer is not None:
                writer.write(frame)
            cv2.imshow("Object detection and tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = max(time.time() - started, 1e-6)
    print(f"\nProcessed {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS on average).")
    for name, ids in sorted(ids_per_class.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name}: {len(ids)} unique object(s)")
    if writer is not None:
        print("Annotated video saved to output/result.mp4")


if __name__ == "__main__":
    main()