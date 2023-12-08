from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO("yolov8n.engine")

    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(not ret):
            continue

        cv2.imshow('f', frame)
        result = model(frame)
        print(result)
        cv2.imshow('ff', result[0].plot())
        cv2.waitKey(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
