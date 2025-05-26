import cv2
from collections import deque
from angle_utils import process_frame_for_angle
from angle_utils import get_angle
from angle_utils import change_color
from angle_utils import smooth
from angle_utils import compare_coordinates
from angle_utils import is_arm_direction

r_angle_buffer = deque(maxlen=10)
l_angle_buffer = deque(maxlen=10)
buffer1 = deque(maxlen=10)
max = 0
min = 50

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
        #右ひじの角度を計算する（身体の各部位には番号がふられている）
        r_current_angle = get_angle(pose_landmark,12,14,16)  # 右肘の角度
        change_color(processed_frame, pose_landmark, 12, 14, (0, 255, 0))  # 線の色を緑に変更
        change_color(processed_frame, pose_landmark, 14, 16, (0, 255, 0)) 
        #左ひじの角度を計算する
        l_current_angle = get_angle(pose_landmark,11,13,15)  # 左肘の角度
        change_color(processed_frame, pose_landmark, 11, 13, (255, 0, 0))
        change_color(processed_frame, pose_landmark, 13, 15, (255, 0, 0)) 
        
        compare_result = compare_coordinates(pose_landmark, 14, 16) #右手首と右肘のx座標を比較 
        if compare_result%10 == 0:
            cv2.putText(frame, "Right wrist is bent", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if compare_result%10 == 1:
            cv2.putText(frame, "Right wrist is up", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if compare_result//10 == 0:
            cv2.putText(frame, "Right wrist is left", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if compare_result//10 == 1:
            cv2.putText(frame, "Right wrist is right", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        

        #腕が前に突き出ているかどうかを確認する
        angle_arm_r = is_arm_direction(pose_landmark, "x")
        smooth_arm_r0 = smooth(buffer1, angle_arm_r)
        cv2.putText(frame, f"Arm direction: {smooth_arm_r0}",
                     (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if smooth_arm_r0 > max:
            max = smooth_arm_r0
        if smooth_arm_r0 < min:
            min = smooth_arm_r0

        # 可視性を取得
        r_visibility = round(pose_landmark.landmark[12].visibility, 2) # 右肩の可視性
        l_visibility = round(pose_landmark.landmark[11].visibility, 2) # 左肩の可視性
    else:
        r_current_angle = None
        l_current_angle = None
        r_visibility = "N/A"
        l_visibility = "N/A"

    #角度のブレ補正
    r_angle = smooth(r_angle_buffer, r_current_angle)
    l_angle = smooth(l_angle_buffer, l_current_angle)

    

    # 結果を画像に描画(x,y)
    cv2.putText(frame, f"Right Elbow Angle: {r_angle} deg, {r_visibility}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Left Elbow Angle: {l_angle} deg, {l_visibility}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    
    # 画面表示
    cv2.imshow("Elbow Angle Detection", processed_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Max:", max)
print("Min:", min)
cap.release()
cv2.destroyAllWindows()
