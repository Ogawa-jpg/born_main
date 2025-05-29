import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from angle_utils import process_frame_for_angle

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
previous_frame = None
check = 0
move_check = deque(maxlen=10)

cap = cv2.VideoCapture(0)

# 背景色（BGR）を設定（例：青）
background_color = (255, 0, 0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR→RGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # セグメンテーション処理
    result = segmentation.process(rgb_frame)

    # マスク取得（0〜1のfloat、shape: (height, width)）
    mask = result.segmentation_mask

    # マスクを0か1に二値化（閾値0.1、調整可能）
    condition = mask > 0.1

    # 背景色と元画像を合成
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    bg_image[:] = background_color

    output_image = np.where(condition[:, :, None], frame, bg_image)

    processed_frame = output_image
    if processed_frame is not None:
        if previous_frame is not None:
            diff = cv2.absdiff(processed_frame, previous_frame)
            diff_sum = np.sum(diff)
            if diff_sum > 2000000:  # 適切な閾値を設定
                check = 1
            else:
                check = 0
        move_check.append(check)
        if len(move_check) == move_check.maxlen:
            if sum(move_check) > 5:
                print("MOVE")
        previous_frame = processed_frame.copy()

    cv2.imshow("Background Replacement", processed_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
