import mediapipe as mp
import numpy as np
import cv2
import math


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def process_frame_for_angle(frame):
    """1フレームから右ひじの角度を計算し、骨格を描画した画像を返す"""
    image_rgb = frame[:, :, ::-1]  # BGR → RGB

    results = pose.process(image_rgb)
    
    # 初期化
    pose_landmark = None

    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        pose_landmark = results.pose_landmarks

    return frame, pose_landmark

# ランドマークの座標を取得する関数
def get_landmark_coordinates(pose_landmark, landmark_index):
    if pose_landmark is not None:
        landmark = pose_landmark.landmark[landmark_index]
        return landmark
    else:
        return None
    
#ランドマーク間の長さを計算する関数
def get_length(pose_landmark, a, b):
    # 引数はランドマークのインデックス
    landmark1 = get_landmark_coordinates(pose_landmark, a)
    landmark2 = get_landmark_coordinates(pose_landmark, b)
    
    if landmark1 is not None and landmark2 is not None:
        length = np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)
        return length
    else:
        return None

# ランドマークのインデックスを指定して、色を変更する関数
def change_color(frame, pose_landmark, a, b, color):
    # 引数はランドマークのインデックス
    landmark1 = get_landmark_coordinates(pose_landmark, a)
    landmark2 = get_landmark_coordinates(pose_landmark, b)
    # 右ひじ → 右手首
    cv2.line(frame, 
            (int(landmark1.x * frame.shape[1]), int(landmark1.y * frame.shape[0])), 
            (int(landmark2.x * frame.shape[1]), int(landmark2.y * frame.shape[0])), 
             color, 4)  

# 3つのランドマークから角度を計算する関数
def get_angle(pose_landmark,a, b, c):
    # 引数はランドマークのインデックス
    landmark1 = get_landmark_coordinates(pose_landmark, a)
    landmark2 = get_landmark_coordinates(pose_landmark, b)  
    landmark3 = get_landmark_coordinates(pose_landmark, c)

    #3つのランドマークから角度を計算する
    a = np.array([landmark1.x, landmark1.y])
    b = np.array([landmark2.x, landmark2.y]) #基準点
    c = np.array([landmark3.x, landmark3.y])

    ab = a - b
    bc = c - b

    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

# 平均化することでブレを軽減する関数
def smooth(buffer, Value):
    if Value is not None:
        buffer.append(Value)
        return round(sum(buffer) / len(buffer))
    else:
        return "N/A"
    
# 座標比較関数
def compare_coordinates(pose_landmark, a, b):
    # 引数はランドマークのインデックス
    landmark1 = get_landmark_coordinates(pose_landmark, a)
    landmark2 = get_landmark_coordinates(pose_landmark, b)
    x = 0

    # 座標を比較(bがaより00は左下,01は左上,10は右下,11は右上)
    if landmark1.x >  landmark2.x:
        x += 10
    if landmark1.y > landmark2.y:
        x +=1
    return x

#カメラに向かって腕を突き出す → 肩から肘、肘から手首の長さが大きく変化する(if rate_arm_r < 1.4 and rate_arm_r > 0.6:,rate_arm_r = get_length(pose_landmark, 12, 14) / get_length(pose_landmark, 14, 16))
#Z座標はカメラに近いほどマイナスになる
#腕が前につき出ているかどうかを確認する関数
def is_arm_direction(pose_landmark, vector):
    # 引数はランドマークのインデックス
    shoulder = get_landmark_coordinates(pose_landmark, 12)  # 右肩
    elbow = get_landmark_coordinates(pose_landmark, 14)  # 右肘
    wrist = get_landmark_coordinates(pose_landmark, 16) # 右手首

    if vector == "x":
        base_vector = np.array([-1, 0, 0])  # x軸方向
    elif vector == "y":
        base_vector = np.array([0, 1, 0]) # y軸方向
    elif vector == "z":
        base_vector = np.array([0, 0, -1])  # z軸方向
    else:
        raise ValueError("Invalid vector. Choose 'x', 'y', or 'z'.")
    
    # 腕の方向ベクトル（肩から肘のベクトル）
    shoulder_to_elbow = (elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z)
    # ベクトルの内積を計算して角度を求める
    dot_product = shoulder_to_elbow[0] * base_vector[0] + shoulder_to_elbow[1] * base_vector[1] + shoulder_to_elbow[2] * base_vector[2]
    magnitude_1 = math.sqrt(shoulder_to_elbow[0]**2 + shoulder_to_elbow[1]**2 + shoulder_to_elbow[2]**2)
    magnitude_2 = math.sqrt(base_vector[0]**2 + base_vector[1]**2 + base_vector[2]**2)

    if magnitude_1 == 0 or magnitude_2 == 0:
        return None
    # 角度（ラジアン）の計算
    cosine_angle = dot_product / (magnitude_1 * magnitude_2)
    angle_rad = math.acos(np.clip(cosine_angle,-1.0, 1.0))

    # ラジアンを度に変換
    angle_deg = math.degrees(angle_rad)    
    
    return angle_deg


def is_significant_change(prev_angles, new_angles, threshold):
    return any(abs(prev_angles[i] - new_angles[i]) > threshold for i in range(6))