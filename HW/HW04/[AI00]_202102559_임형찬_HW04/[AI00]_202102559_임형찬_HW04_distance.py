from ultralytics import YOLO
import cv2
import numpy as np
import math

if __name__=='__main__':
    model = YOLO('best(3).pt')

    cap = cv2.VideoCapture(0)
    box_history = []

    prev_boxes = []
    global_idx = 0 # id
    match_dist_threshold = 300 # id 유지를 위한 거리 threshold

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            res = model(frame)

            current_boxes = []
            for box in res[0].boxes:
                box = box.cpu() # GPU를 사용했다면 cpu에서 연산하기 위해서 사용
                if int(box.cls) == 0: # box가 hand인지 확인
                    box_info = np.array(box.xyxy, dtype=int)[0]
                    center = [int((box_info[0] + box_info[2]) / 2), int((box_info[1] + box_info[3]) / 2)] # 현재 frame의 가운데 지점
                    current_boxes.append(center)

            # TODO: Match by distance
            # 현재 frame의 box1의 중앙과 이전 frame의 box들의 중앙간의 거리 계산
            # 계산한 거리가 이전 frame의 여러 박스들 중 가장 가깝고, match_dist_threshold보다 작은 것과 매칭
            # 만약 위 두 조건을 만족한다면 id 유지, 만족하지 못하면 id+1

            res_plot = res[0].plot()

            for c_box in current_boxes:
                best_dist = float('inf')
                # best_dist = 0도 해보고 float('inf')도 해봤는데 0으로 했을때는 숫자가 너무 급격하게 올라감.
                # 0으로 한 이유는 이전 IoU코드를 baseline 코드로 잡고 시작했기 때문.
                # 따라서 구글링을 하던 중 inf로 시작하길래 나도 inf로 잡고 시작함.

                # 이유를 알아보니 best_dist를 초기에 무한대로 설정해야 처음계산하는 거리가 항상 best_dist초기값보다 작도록 보장되기 때문
                # 0으로 설정하면 처음거리를 최선거리로 취급하고 거리비교동안 잘못된 일치가 발생할 확률이 있다고함.
                best_match_id = -1

                # 이전 프레임의 박스들을 반복하는데, 이때 enumerate 함수를 사용하여
                #  각 박스와 해당 박스의 인덱스를 함께 반복시켜준다.
                for i, p_center in enumerate(prev_boxes):
                    # numpy에 Norm_Distance함수를 이용하면 쉽게 거리를 구할 수 있음.
                    dist = np.linalg.norm(np.array(c_box) - np.array(p_center))

                    # 현재 계산한 거리 dist가 현재까지의 best_dist보다 작고,
                    # 그와 동시에 match_dist_threshold보다 작은 경우 검사를 진행.
                    if dist < best_dist and dist < match_dist_threshold:
                        best_dist = dist
                        best_match_id = i

                # 여기서부터는 IoU코드와 동일.
                if best_match_id != -1:
                    c_box_id = global_idx + best_match_id
                else:
                    c_box_id = global_idx
                    global_idx += 1
                cv2.putText(res_plot, f"{c_box_id}", (c_box[0], c_box[1]), 0, 2, (0, 255, 0), 1, 1)

            prev_boxes = current_boxes

            cv2.imshow('f', res_plot)
            k = cv2.waitKey(1)

            if k == ord('q'):
                break
