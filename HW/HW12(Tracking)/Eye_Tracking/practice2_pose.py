from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8s-pose.pt')

cap = cv2.VideoCapture(0)
le_history = []
re_history = []
history_len = 20
# "keypoints": [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        res = model(frame)

        for kp in res[0].keypoints:
            kp = kp.cpu()
            	    
            left_eye = np.array(kp.data[0, 1, :2], dtype=int)
            right_eye = np.array(kp.data[0, 2, :2], dtype=int)

            le_history.append(left_eye)
            re_history.append(right_eye)

            if len(le_history) > history_len:
                le_history = le_history[-history_len:]
            if len(re_history) > history_len:
                re_history = re_history[-history_len:]

        # res_plot = res[0].plot()
        res_plot = frame

        for c in le_history:
            cv2.circle(res_plot, c, 2, (0, 0, 255), -1)

        for c in re_history:
            cv2.circle(res_plot, c, 2, (255, 255, 0), -1)

        cv2.imshow('f', res_plot)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break
