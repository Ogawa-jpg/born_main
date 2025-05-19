import cv2
from collections import deque
from angle_utils import process_frame_for_angle
from angle_utils import get_angle
from angle_utils import change_color
from angle_utils import smooth_angle
from angle_utils import compare_coordinates



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
        change_color(processed_frame, pose_landmark, 11, 13, (255, 0, 0))  # 線の色を緑に変更
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
        
        # 可視性を取得
        visibility = round(pose_landmark.landmark[12].visibility, 2)  # 右肘の可視性
    else:
        r_current_angle = None
        l_current_angle = None
        visibility = "N/A"

    r_angle_buffer = deque(maxlen=10)
    r_angle = smooth_angle(r_angle_buffer, r_current_angle)
    l_angle_buffer = deque(maxlen=10)
    l_angle = smooth_angle(l_angle_buffer, l_current_angle)

    

    # 結果を画像に描画(x,y)
    cv2.putText(frame, f"Right Elbow Angle: {r_angle} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Left Elbow Angle: {l_angle} deg",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    
    # 画面表示
    cv2.imshow("Elbow Angle Detection", processed_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
