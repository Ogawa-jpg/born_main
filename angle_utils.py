import mediapipe as mp
import numpy as np
import cv2


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
    b = np.array([landmark2.x, landmark2.y])
    c = np.array([landmark3.x, landmark3.y])

    ab = a - b
    bc = c - b

    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def smooth_angle(angle_buffer, current_angle):
    """角度をスムーズにするための関数"""
    if current_angle is not None:
        angle_buffer.append(current_angle)
        return round(sum(angle_buffer) / len(angle_buffer))
    else:
        return "N/A"