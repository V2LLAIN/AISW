from ultralytics import YOLO
import cv2
import numpy as np
import math

def get_IoU(box1, box2):
    # box = (x1, y1, x2, y2) == (xmin, ymin, xmax, ymax)
    # fill with a function created in practice time
    iou = 0
    # TODO : write a function that calcuates IoU
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

if __name__=='__main__':
    model = YOLO('best(3).pt')

    cap = cv2.VideoCapture(0)
    box_history = []

    prev_boxes = []
    global_idx = 0 # id
    match_IoU_threshold = 0.5 # id 유지를 위한 iou threshold

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            res = model(frame)

            current_boxes = []
            for box in res[0].boxes:
                box = box.cpu() # GPU를 사용했다면 cpu에서 연산하기 위해서 사용
                if int(box.cls) == 0: # box가 hand인지 확인
                    box_info = np.array(box.xyxy, dtype=int)[0]
                    box_info_tolist = [xy for xy in box_info]
                    current_boxes.append(box_info_tolist)

            # TODO: Match by IoU
            # 현재 frame box1과 이전 frame의 box들과 iou 계산
            # iou가 가장 크고, match_IoU_threshold보다 iou가 큰 것과 매칭
            # 위 조건을 만족하면 id 유지, 만족하지 못하면 id+1

            res_plot = res[0].plot()

            for c_box in current_boxes:
                # 현재까지 찾은 최선의 IoU와 해당 idx를 초기화
                best_iou = 0
                best_match_idx = -1

                # 이전 프레임의 각 박스를 돌며 enumerate를 이용해 각 박스와 해당 인덱스를 함께 구간반복.
                for i, pre_box in enumerate(prev_boxes):
                    iou = get_IoU(c_box, pre_box) # 이전과 현재 박스간의 iou값을 계산

                # 미리 설정한 threshold를 이용해 임계값을 넘어야 실행되도록함.
                    if iou > best_iou and iou > match_IoU_threshold:
                        best_iou = iou
                        best_match_idx = i

                if best_match_idx != -1:
                    c_box_id = global_idx + best_match_idx
                else:
                    c_box_id = global_idx
                    global_idx += 1
                cv2.putText(res_plot, f"{c_box_id}", (c_box[0], c_box[1]), 0, 2, (0, 255, 0), 1, 1)

            prev_boxes = current_boxes

            cv2.imshow('f', res_plot)
            k = cv2.waitKey(1)

            if k == ord('q'):
                break

