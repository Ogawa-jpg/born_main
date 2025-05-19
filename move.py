import cv2
from angle_utils import process_frame_for_angle
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

    # リアルタイムで角度計算
    processed_frame, current_angle, visibility = process_frame_for_angle(frame)

    if current_angle is not None:
        angle_buffer.append(current_angle)
        angle = round(sum(angle_buffer) / len(angle_buffer))
    else:
        angle = "N/A"



    # 結果を画像に描画(x,y)
    cv2.putText(frame, f"Right Elbow Angle: {angle} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Visibility: {visibility['right_elbow']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
    
    # 画面表示
    cv2.imshow("Elbow Angle Detection", processed_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
