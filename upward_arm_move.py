import threading ,queue ,time, cv2
from collections import deque
from pymycobot.mycobot280 import MyCobot280
from angle_utils import process_frame_for_angle, get_angle, is_arm_direction
from angle_utils import is_significant_change, Collision_prevention, compare_coordinates, filter_angle
from angle_utils import filter_movement


# MyCobot初期化
mc = MyCobot280('COM4', 115200)
speed = 100
mc.set_color(0, 0, 255)  # 緑色に設定

# 角度のフィルタリングのためのデバッファ
angle_buffer = deque(maxlen=2)  # 過去2フレームの角度を保持
# 位置の履歴
wrist_history = deque(maxlen=2)  # 手首位置の履歴を保持


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

cap = cv2.VideoCapture(0)
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
            r_current_angle = get_angle(pose_landmark, 12, 14, 16)
            r_filter_angle = filter_angle(r_current_angle, angle_buffer, 90)  # 90°より大きい角度変化を受け付けないようにフィルタリング
            elbow_angle = -(180 - r_filter_angle) * (150 / 180)
            if compare_coordinates(pose_landmark, 14, 16)%10 == 0:  #右手首が右肘よりも下側にある場合
                elbow_angle = -elbow_angle
            
            sholder_angle_x = is_arm_direction(pose_landmark, "x")
            if sholder_angle_x <= 75:
                sholder_angle_x = 90
            elif sholder_angle_x > 75:
                sholder_angle_x = 0

            sholder_angle_y = 160 - is_arm_direction(pose_landmark, "y")
            if sholder_angle_y > 135:
                sholder_angle_y = 135
            


            # 肘の角度を調整
            elbow_angle = Collision_prevention(margin=10, a_deg=sholder_angle_y, b_deg=elbow_angle)
            new_angles = [sholder_angle_x, sholder_angle_y, elbow_angle, 0, 0, 0]

            # 角度が大きく変わった時だけ送るなどの条件も入れられる
            if prev_angles is None or is_significant_change(prev_angles, new_angles, 30):
                wrist_movement = filter_movement(pose_landmark, 16 ,wrist_history)
                if wrist_movement is False: # 手首の動きが小さい場合は角度を送信しない
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
