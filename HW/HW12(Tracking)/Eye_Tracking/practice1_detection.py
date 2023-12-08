from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8s.pt')
# model = YOLO('yolov8s-seg.pt')

cap = cv2.VideoCapture(0)
box_history = []
history_len = 100

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        res = model(frame)

        for box in res[0].boxes:
            box = box.cpu()
            if int(box.cls) == 0:
                box_info = np.array(box.xyxy, dtype=int)[0]
                center = (int((box_info[0] + box_info[2]) / 2), int((box_info[1] + box_info[3]) / 2))
                box_history.append(center)
                if len(box_history) > history_len:
                    box_history = box_history[-history_len:]

        res_plot = res[0].plot()

        for c in box_history:
            cv2.circle(res_plot, c, 2, (255, 255, 0), -1)

        cv2.imshow('f', res_plot)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break
