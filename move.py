import cv2
from angle_utils import process_frame_for_angle
from angle_utils import get_angle
from angle_utils import get_landmark_coordinates
from angle_utils import change_color
from collections import deque

angle_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)  # 0番は通常、内蔵または最初に見つかるカメラ

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #骨格検出を行う
    processed_frame, pose_landmark = process_frame_for_angle(frame)
    if pose_landmark is not None:
        #角度を計算する
        current_angle = get_angle(pose_landmark,12,14,16)  # 右肘の角度
        change_color(processed_frame, pose_landmark, 12, 14, (0, 255, 0))  # 線の色を緑に変更
        change_color(processed_frame, pose_landmark, 14, 16, (0, 255, 0)) 

        # 可視性を取得
        if pose_landmark is not None:
            visibility = round(pose_landmark.landmark[12].visibility, 2)  # 右肘の可視性
        else:
            visibility = "N/A"
    else:
        current_angle = None
        visibility = "N/A"

    if current_angle is not None:
        angle_buffer.append(current_angle)
        angle = round(sum(angle_buffer) / len(angle_buffer))
    else:
        angle = "N/A"

    # 結果を画像に描画(x,y)
    cv2.putText(frame, f"Right Elbow Angle: {angle} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Visibility: {visibility}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    
    # 画面表示
    cv2.imshow("Elbow Angle Detection", processed_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
