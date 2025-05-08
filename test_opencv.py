import cv2
import sys
import time

print(f"使用的 OpenCV 版本: {cv2.__version__}")

# 尝试打开默认摄像头 (通常索引为 0)
camera_index = 0
print(f"正在尝试打开摄像头索引: {camera_index}")
cap = cv2.VideoCapture(camera_index)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print(f"错误：无法打开摄像头索引 {camera_index}。")
    print("请检查：")
    print("1. 摄像头是否已连接并被系统识别。")
    print("2. 是否有其他程序正在使用该摄像头。")
    print("3. 摄像头的索引是否正确 (可以尝试 1, 2, ...)。")
    sys.exit()
else:
    print(f"摄像头 {camera_index} 成功打开。")

window_name = "OpenCV 测试窗口 (按 'q' 退出)"
print(f"正在创建窗口: '{window_name}'")
# 预先创建窗口，有时有助于避免问题
try:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("窗口创建成功。")
except Exception as e:
    print(f"创建窗口时出错: {e}")
    cap.release()
    sys.exit()

print("开始读取和显示帧... 按 'q' 键退出循环。")

frame_count = 0
start_time = time.time()

while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("错误：无法从摄像头读取帧。可能是摄像头断开连接。")
        break

    # 显示帧
    try:
        cv2.imshow(window_name, frame)
        frame_count += 1
    except Exception as e:
        print(f"\n调用 cv2.imshow 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        break

    # 等待按键事件 (等待 1 毫秒)
    # & 0xFF 是为了确保在不同系统上获得一致的按键码
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 'q' 键，则退出循环
    if key == ord('q'):
        print("\n检测到 'q' 键按下，正在退出...")
        break

# 循环结束后，释放摄像头资源并销毁所有 OpenCV 窗口
print("正在释放摄像头资源...")
cap.release()
print("正在销毁所有 OpenCV 窗口...")
cv2.destroyAllWindows()
# 在某些系统上，可能需要额外等待一小段时间确保窗口完全关闭
# cv2.waitKey(1)

end_time = time.time()
duration = end_time - start_time
fps = frame_count / duration if duration > 0 else 0
print(f"测试结束。总帧数: {frame_count}, 持续时间: {duration:.2f} 秒, 平均 FPS: {fps:.2f}")