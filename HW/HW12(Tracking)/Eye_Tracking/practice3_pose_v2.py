from ultralytics import YOLO
import cv2
import numpy as np

fire = cv2.imread('fire.png')
fire = cv2.resize(fire, None, fx=0.1, fy=0.1)
gray = cv2.cvtColor(fire, cv2.COLOR_BGR2GRAY)

mask = np.where(gray != 0)
gray[mask] = 255

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

        # res_plot = res[0].plot()
        res_plot = frame

        for kp in res[0].keypoints:
            kp = kp.cpu()
            # [num_person, kp idx, data] [1, 17, 3] if there is only one person.

            #left_shoulder = np.array(kp[0, 5, :].data[:2], dtype=int)
            #right_shoulder = np.array(kp[0, 6, :].data[:2], dtype=int)
            left_shoulder = np.array(kp.data[0, 5, :], dtype=int)
            right_shoulder = np.array(kp.data[0, 6, :], dtype=int)


            # left_shoulder = np.array(kp[0, 9, :].data[:2], dtype=int) # left wrist
            # right_shoulder = np.array(kp[0, 10, :].data[:2], dtype=int) # right wrist

            if left_shoulder[1] - fire.shape[0] > 0 and left_shoulder[0] - fire.shape[1] > 0:
                res_plot[left_shoulder[1] - fire.shape[0]:left_shoulder[1], left_shoulder[0] - fire.shape[1]:left_shoulder[0], :][mask] = fire[mask]
            if right_shoulder[1] - fire.shape[0] > 0 and right_shoulder[0] - fire.shape[1] > 0:
                res_plot[right_shoulder[1] - fire.shape[0]:right_shoulder[1], right_shoulder[0] - fire.shape[1]:right_shoulder[0], :][mask] = fire[mask]

        cv2.imshow('f', res_plot)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break
