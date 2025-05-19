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
    visibility = {
            'right_shoulder': 0,
            'right_elbow': 0,
            'right_wrist': 0
        }
    angle = None
    
    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # 右肩、右ひじ、右手首の接続部分をカスタマイズ
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # 右肩 → 右ひじ
        cv2.line(frame, 
                 (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), 
                 (int(right_elbow.x * frame.shape[1]), int(right_elbow.y * frame.shape[0])), 
                 (0, 255, 0), 4)  # 緑色の線

        # 右ひじ → 右手首
        cv2.line(frame, 
                 (int(right_elbow.x * frame.shape[1]), int(right_elbow.y * frame.shape[0])), 
                 (int(right_wrist.x * frame.shape[1]), int(right_wrist.y * frame.shape[0])), 
                 (0, 255, 0), 4)  # 緑色の線

        # 角度計算
        vec_a = np.array([right_shoulder.x, right_shoulder.y]) - np.array([right_elbow.x, right_elbow.y])
        vec_c = np.array([right_wrist.x, right_wrist.y]) - np.array([right_elbow.x, right_elbow.y])
        cos = np.inner(vec_a, vec_c) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        angle = round(np.degrees(rad), 2)

        visibility = {
            'right_shoulder': round(right_shoulder.visibility,2),
            'right_elbow': round(right_elbow.visibility,2),
            'right_wrist': round(right_wrist.visibility,2)
       }
    return frame, angle, visibility
