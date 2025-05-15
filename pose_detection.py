import cv2
import mediapipe as mp
import numpy as np

# 画像ファイルのパス
IMAGE_PATH = "C:/NT_Kanazawa/born/thao-lee-v4zceVZ5HK8-unsplash.jpg" # 任意の画像ファイル名に変更

# MediaPipe Pose モジュールの準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 画像の読み込み
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"画像が見つかりません: {IMAGE_PATH}")
    exit()

# BGRからRGBに変換
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Poseインスタンスで画像処理
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)

    # 骨格が検出されたら描画
    if results.pose_landmarks:
        # ランドマークの点（太さや色を変更）
        landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=20)
        # 接続線のスタイル（任意）
        connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5)

        # 骨格を描画
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )
        # 関節位置の例（右肩）
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        print(f"右肩の座標: x={right_shoulder.x:.3f}, y={right_shoulder.y:.3f}, z={right_shoulder.z:.3f}")
        # 関節位置の例（右肘）
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        print(f"右肘の座標: x={right_elbow.x:.3f}, y={right_elbow.y:.3f}, z={right_elbow.z:.3f}")
        # 関節位置の例（右手首）
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        print(f"右手首の座標: x={right_wrist.x:.3f}, y={right_wrist.y:.3f}, z={right_wrist.z:.3f}")
    else:
        print("骨格を検出できませんでした。")


#角度
Sholuder = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
Elbow = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
Wrist = np.array([right_wrist.x, right_wrist.y, right_wrist.z])

vec_a = Sholuder - Elbow
vec_c = Wrist - Elbow

length_vec_a = np.linalg.norm(vec_a)
length_vec_c = np.linalg.norm(vec_c)
inner_product = np.inner(vec_a, vec_c)
cos = inner_product / (length_vec_a * length_vec_c)

# 角度（ラジアン）の計算
rad = np.arccos(cos)


# 弧度法から度数法（rad ➔ 度）への変換
R_degree = np.rad2deg(rad)

R_degree = round(R_degree, 2)
print('右ひじの角度:',R_degree)


# 画像の縦横比を保ってリサイズして表示
max_width = 800
max_height = 600
h, w = image.shape[:2]
scale = min(max_width / w, max_height / h)
new_w = int(w * scale)
new_h = int(h * scale)
display_image = cv2.resize(image, (new_w, new_h))
cv2.imshow('Pose Detection Result', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
