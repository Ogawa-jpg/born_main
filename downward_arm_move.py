from pymycobot.mycobot280 import MyCobot280
import cv2
import threading
import queue
from collections import deque
from angle_utils import process_frame_for_angle
from angle_utils import get_angle
from angle_utils import change_color
from angle_utils import smooth
from angle_utils import compare_coordinates
from angle_utils import is_arm_direction
from angle_utils import is_significant_change
import time

mc = MyCobot280('COM4',115200)
speed = 50
prev_time = time.time()
flag = 0

cap = cv2.VideoCapture(0)  # 0番は通常、内蔵または最初に見つかるカメラ
mc.set_color(255, 0, 0)

# 角度送信用のスレッドセーフなキュー
angle_queue = queue.Queue()

def robot_control_worker():
    prev_angles = None  # 初期角度
    while True:
        angles = angle_queue.get()  # キューから角度を取得（待機）
        if angles is None:
            break  # Noneが来たら終了
        
        mc.send_angles(angles, speed)
        prev_angles = angles
    time.sleep(0.2)  # 少し待機してから次の角度を取得
       

# 制御スレッド開始
control_thread = threading.Thread(target=robot_control_worker)
control_thread.start()

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

try:
    prev_angles = [0, 0, 0, 0, 0, 0]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, pose_landmark = process_frame_for_angle(frame)
        if pose_landmark is not None:
            #右ひじの角度を計算する（身体の各部位には番号がふられている）
            r_current_angle = get_angle(pose_landmark,12,14,16)  # 右肘の角度
            
            
            elbow_angle = (180 - r_current_angle)*(150/180)
            if compare_coordinates(pose_landmark, 14, 16)%10 == 0:  #右手首が右肘よりも下側にある場合
                elbow_angle = -elbow_angle
                flag = 1

            #腕の方向を得る
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

            flag = 0

            new_angles = [sholder_angle_x, sholder_angle_y, elbow_angle, 0, 0, 0]

            # 角度が大きく変わった時だけ送るなどの条件も入れられる
            if prev_angles is None or is_significant_change(prev_angles, new_angles, 30):
                angle_queue.put(new_angles)
                prev_angles = new_angles

        cv2.imshow("Elbow Angle Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            angle_queue.put([0,0,0,0,0,0],50)  # 終了シグナルを送る
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    mc.set_color(0, 0, 255)

    # 終了シグナルを送ってスレッド停止
    angle_queue.put(None)
    control_thread.join()
