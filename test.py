from pymycobot.mycobot280 import MyCobot280
import cv2
from collections import deque
from angle_utils import process_frame_for_angle
from angle_utils import get_angle
from angle_utils import change_color
from angle_utils import smooth
from angle_utils import compare_coordinates
from angle_utils import is_arm_stretched

mc = MyCobot280('COM4',115200)

cap = cv2.VideoCapture(0)  # 0番は通常、内蔵または最初に見つかるカメラ

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, pose_landmark = process_frame_for_angle(frame)
    if pose_landmark is not None:
        #右ひじの角度を計算する（身体の各部位には番号がふられている）
        r_current_angle = get_angle(pose_landmark,12,14,16)  # 右肘の角度
    
    # 画面表示
    cv2.imshow("Elbow Angle Detection", processed_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#angle = mc.get_angles()
#print("Angle :",angle)

mc.set_color(0, 255, 0)
#[根本: ~ ~ : 頂点]
#mc.send_angles([0, 0, 0, 0, 0, 0], 50)
