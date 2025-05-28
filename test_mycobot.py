from pymycobot.mycobot280 import MyCobot280
from angle_utils import calculate_arm_height, Collision_prevention
import time

mc = MyCobot280('COM4', 115200)
mc.set_color(0, 255, 0)  # 緑色に設定
mc.send_angles([0, 0, 0, 0, 0, 0], 50)  
time.sleep(2)  # アームが動くのを待つ

a = mc.get_angles()[1]  # アームの角度を取得
b = mc.get_angles()[2]  # アームの角度を取得
print(f"アームの角度: {a}, {b}")

h = calculate_arm_height(a, b)  # アームの高さを計算
b = Collision_prevention(margin=3, a_deg=a, b_deg=b)
mc.send_angles([0, a, b, 0, 0, 0], 50)
     

print(f"アームの高さ: {h:.2f} cm")

    