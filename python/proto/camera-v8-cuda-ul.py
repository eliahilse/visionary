import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('yolov8x.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cam error -> couldn't get frame (!)")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()