import cv2
import numpy as np
from sort import Sort

"""
Detect object without YOLO and track with SORT
"""

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"Sample.mp4")

prev_frame_time = 0
new_frame_time = 0

object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40)
mot_tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            dets.append([x, y, x+w, y+h, 1.0])
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
    dets = np.array(dets)
    trackers = mot_tracker.update(dets)

    for d in trackers:
        x1, y1, x2, y2, track_id = map(int, d)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2)

    cv2.imshow("view", frame)
    # cv2.imshow("mask", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


