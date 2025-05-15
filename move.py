import cv2
from angle_utils import process_frame_for_angle

cap = cv2.VideoCapture(0)  # 0番は通常、内蔵または最初に見つかるカメラ

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # リアルタイムで角度計算
    processed_frame, angle = process_frame_for_angle(frame)

    # 画面表示
    cv2.imshow("Elbow Angle Detection", processed_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
