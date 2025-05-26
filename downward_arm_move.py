from pymycobot.mycobot280 import MyCobot280
import cv2
from collections import deque
from angle_utils import process_frame_for_angle
from angle_utils import get_angle
from angle_utils import change_color
from angle_utils import smooth
from angle_utils import compare_coordinates
from angle_utils import is_arm_direction
import time

mc = MyCobot280('COM4',115200)
prev_time = time.time()
flag = 0

cap = cv2.VideoCapture(0)  # 0番は通常、内蔵または最初に見つかるカメラ

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    mc.set_color(255, 0, 0)
    processed_frame, pose_landmark = process_frame_for_angle(frame)
    if pose_landmark is not None:
   
        current_time = time.time()
        if  current_time - prev_time > 0.5:
            #右ひじの角度を計算する（身体の各部位には番号がふられている）
            r_current_angle = get_angle(pose_landmark,12,14,16)  # 右肘の角度
            
            print("Right Elbow Angle:", r_current_angle)
            elbow_angle = (180 - r_current_angle)*(150/180)
            if compare_coordinates(pose_landmark, 14, 16)%10 == 0:  #右手首が右肘よりも下側にある場合
                elbow_angle = -elbow_angle
                flag = 1

            #腕の方向をえる
            sholder_angle_x = is_arm_direction(pose_landmark, "x")
            if sholder_angle_x <= 75:
                sholder_angle_x = 90
            elif sholder_angle_x > 75:
                sholder_angle_x = 0

            if compare_coordinates(pose_landmark, 12, 14)//10 == 0: 
                sholder_angle_x = -90
                if flag == 0:
                    elbow_angle = -elbow_angle
                    flag = 1

            sholder_angle_y = is_arm_direction(pose_landmark, "y")
            if sholder_angle_y > 135:
                sholder_angle_y = 135
            

            mc.send_angles([sholder_angle_x, sholder_angle_y, elbow_angle, 0, 0, 0], 50)
            prev_time = current_time
            flag = 0

    # 画面表示
    cv2.imshow("Elbow Angle Detection", processed_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#angle = mc.get_angles()
#print("Angle :",angle)

mc.set_color(0, 0, 255)
#[根本: ~ ~ : 頂点]send_angles(degrees, speed) : 複数関節の角度(度)の変更
#degree : 関節の角度のリスト ([float], 0は-168～168、1~5は-150~150でまちまち,  長さ6)
#speed : 速度 (int, 0~100)
mc.send_angles([0, 0, 0, 0, 0, 0], 50)
