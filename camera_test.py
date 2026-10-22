import cv2

# 0 = the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Cannot open the webcam.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam opened: {width}x{height}. Press 'q' to quit.")

while True:
    ok, frame = cap.read()  # read one frame
    if not ok:
        print("Cannot read a frame from the webcam.")
        break

    cv2.imshow("Webcam test", frame)  # show it in a window

    # wait 1 ms for a key; stop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()