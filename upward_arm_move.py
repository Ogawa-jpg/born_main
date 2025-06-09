import threading ,queue ,time, cv2
from collections import deque
from pymycobot.mycobot280 import MyCobot280
from angle_utils import process_frame_for_angle, get_angle, is_arm_direction
from angle_utils import is_significant_change, Collision_prevention, compare_coordinates, filter_angle
from angle_utils import filter_movement, Collision_prevention_with_table


# MyCobot初期化
mc = MyCobot280('COM4', 115200)
speed = 100
mc.set_color(0, 0, 255)  # 緑色に設定

## 過去2フレームの角度を保持のためのデバッファ
angle_buffer = deque(maxlen=2)  
# 手首の動きを保存するためのデバッファ
wrist_history = deque(maxlen=2)
# 角度送信用のスレッドセーフなキュー
angle_queue = queue.Queue()
prev_sholder_angle_x = 0  # 前回の肩の角度を保持
prev_move = 0

SEND_INTERVAL = 0.2  # 送信間隔（秒）
last_send_time = time.time()

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
        
            
            # 肩の角度
            sholder_angle_x = is_arm_direction(pose_landmark, "x")
            if sholder_angle_x > 85:
                sholder_angle_x = 0
            elif sholder_angle_x <= 85 and sholder_angle_x >= 70:
                sholder_angle_x = prev_sholder_angle_x
            elif sholder_angle_x < 70:
                sholder_angle_x = 90
            prev_sholder_angle_x = sholder_angle_x
            
            # 肘の角度
            r_current_angle = get_angle(pose_landmark, 12, 14, 16)
            # 腕を前に突き出したときのような角度が正しく検出できなかったときの対策
            r_filter_angle = filter_angle(r_current_angle, angle_buffer, 90)  # 90°より大きい角度変化を受け付けないようにフィルタリング
            elbow_angle = -(180 - r_filter_angle) * (150 / 180)
            if compare_coordinates(pose_landmark, 14, 16)%10 == 0:  #右手首が右肘よりも下側にある場合
                if compare_coordinates(pose_landmark, 12, 16)%10 == 0: #右手首が右肩よりも下側にある場合
                    if sholder_angle_x != 0:  # 肩の角度が0でない場合
                        elbow_angle = -elbow_angle    

            sholder_angle_y = 160 - is_arm_direction(pose_landmark, "y")
            if sholder_angle_y > 135:
                sholder_angle_y = 135
            


            # 安全機構
            # 肩の角度と肘の角度を使って衝突防止のための角度を計算（机の上においてある場合、机の上面との接触を防ぐ）
            #elbow_angle = Collision_prevention(margin=10, a_deg=sholder_angle_y, b_deg=elbow_angle)
            # 机の端に置いてある場合、机の横面との接触を防ぐ
            elbow_angle = Collision_prevention_with_table(a_deg=sholder_angle_y, b_deg=elbow_angle)
            new_angles = [sholder_angle_x, sholder_angle_y, elbow_angle, 0, 0, 0]

            #stopかmoveかを判定
            current_move = filter_movement(pose_landmark, 16, wrist_history, 0.01)
            if current_move == True:
                current_move = 1
            else:
                current_move = 0
            # 前の動きと比較して変化があったかどうかを確認
            differ_move = current_move - prev_move
            if differ_move < 0:
                differ_move = 1
            # 角度が大きく変わった時だけ送る + move → stopの時だけ送る
            # 肘と肩で閾値を変えると同期してる感が出る？
            now = time.time()
            should_send = False

            if differ_move == 1:
                if prev_angles is None or is_significant_change(prev_angles, new_angles, 10):
                    should_send = True
            
            #前に送った時刻と比較して近すぎたら送らない
            if should_send and (now - last_send_time) > SEND_INTERVAL:
                angle_queue.put(new_angles)
                prev_angles = new_angles
                last_send_time = now
            
            
            
            prev_move = current_move

        cv2.imshow("Elbow Angle Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            angle_queue.put([0,0,0,0,0,0])  # 終了シグナルを送る
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    mc.set_color(0, 0, 255)

    # 終了シグナルを送ってスレッド停止
    angle_queue.put(None)
    control_thread.join()
