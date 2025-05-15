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

    angle = None
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # 関節取得
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # 角度計算
        vec_a = np.array([shoulder.x, shoulder.y, shoulder.z]) - np.array([elbow.x, elbow.y, elbow.z])
        vec_c = np.array([wrist.x, wrist.y, wrist.z]) - np.array([elbow.x, elbow.y, elbow.z])
        cos = np.inner(vec_a, vec_c) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        angle = round(np.degrees(rad), 2)

        # 結果を画像に描画
        cv2.putText(frame, f"Right Elbow Angle: {angle} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame, angle
