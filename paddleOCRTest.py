#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


class RealtimeVideoOCR:
    def __init__(self, use_gpu=True, lang="ch", det_model_dir=None, rec_model_dir=None):
        """
        初始化实时视频OCR系统

        Args:
            use_gpu: 是否使用GPU加速
            lang: OCR语言，默认为中文
            det_model_dir: 检测模型目录，默认为None使用PaddleOCR自带模型
            rec_model_dir: 识别模型目录，默认为None使用PaddleOCR自带模型
        """
        # 针对M芯片Mac上的错误进行特殊处理
        # # 初始化PaddleOCR，避免使用不兼容的选项
        # self.ocr = PaddleOCR(
        #     use_angle_cls=False,  # 关闭方向分类，提高速度
        #     use_gpu=use_gpu,
        #     lang=lang,
        #     det_model_dir=det_model_dir,
        #     rec_model_dir=rec_model_dir,
        #     # 使用轻量级检测模型
        #     det_db_thresh=0.3,    # 降低检测阈值
        #     det_db_box_thresh=0.5,  # 降低检测框阈值
        #     det_limit_side_len=960,  # 限制最大边长
        #     # 使用轻量级识别模型
        #     rec_batch_num=30,  # 批处理大小
        #     # 注意：这里移除了enable_mkldnn=True选项，因为它在M芯片上不兼容
        #     # 设置以下选项来提高性能
        #     use_mp=True,  # 使用多进程加速
        #     total_process_num=10  # 只使用1个进程避免额外开销
        # )

        # 初始化视频捕获
        self.cap = None

        # ROI区域，格式为 (x, y, width, height)
        self.roi = (0, 0, 400, 200)

        # 标志位
        self.is_running = False

        # 帧率计算
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0

        # 上次OCR时间，用于控制OCR频率
        self.last_ocr_time = 0
        self.ocr_interval = 0.5  # 增加到1秒，减少处理频率

        # 最近的OCR结果
        self.ocr_results = []

        # 多线程处理标志
        self.processing = False

    def start_capture(self, camera_index=1, resolution=(1280, 720)):
        """
        启动视频捕获

        Args:
            camera_index: 摄像头索引
            resolution: 视频分辨率，默认设为较低的(640, 480)提高性能
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {camera_index}")
            return False

        # 设置较低的分辨率以提高处理速度
        width, height = resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 降低缓冲区大小，减少延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        return True

    def set_roi(self, x, y, width, height):
        """设置ROI区域"""
        self.roi = (x, y, width, height)

    def clear_roi(self):
        """清除ROI区域"""
        self.roi = None

    def process_frame(self, frame):
        """处理单帧图像"""
        display_frame = frame.copy()

        # 确定处理区域
        if self.roi:
            x, y, w, h = self.roi
            # 确保ROI在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)

            # 在原图上绘制ROI区域
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 在图像上绘制OCR结果
        if self.ocr_results:
            for line in self.ocr_results:
                try:
                    # 获取文字框坐标并缩放回原始大小
                    points = np.array(line[0], dtype=np.float32)
                    points = points.astype(np.int32)

                    if self.roi:
                        # 坐标需要加上ROI的偏移
                        points[:, 0] += self.roi[0]  # 加上x偏移
                        points[:, 1] += self.roi[1]  # 加上y偏移

                    # 绘制文字框
                    cv2.polylines(display_frame, [points], True, (0, 0, 255), 2)

                    # 获取文本和置信度
                    text = line[1][0]
                    confidence = line[1][1]

                    print(f"识别结果: {text} ({confidence:.2f})")

                    # 显示文本
                    text_position = (points[0][0], points[0][1] - 10)
                    cv2.putText(
                        display_frame,
                        f"{text} ({confidence:.2f})",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                except Exception as e:
                    print(f"绘制OCR结果出错: {e}")
                    continue

        # 计算并显示FPS
        self.new_frame_time = time.time()
        self.fps = (
            1 / (self.new_frame_time - self.prev_frame_time)
            if self.prev_frame_time > 0
            else 0
        )
        self.prev_frame_time = self.new_frame_time

        # 显示FPS
        cv2.putText(
            display_frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return display_frame

    def run(self):
        """运行OCR系统"""
        if not self.cap or not self.is_running:
            print("摄像头未初始化，请先调用start_capture()")
            return

        print("按 'r' 设置ROI区域")
        print("按 'c' 清除ROI区域")
        print("按 'q' 退出程序")

        selecting_roi = False
        roi_start_point = None

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                break

            if selecting_roi:
                # 绘制临时ROI
                temp_frame = frame.copy()
                if roi_start_point:
                    cv2.rectangle(
                        temp_frame,
                        roi_start_point,
                        (self.mouse_x, self.mouse_y),
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    temp_frame,
                    "正在选择ROI，请拖动鼠标并点击左键确认",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                cv2.imshow("实时视频OCR", temp_frame)
            else:
                # 正常处理帧
                processed_frame = self.process_frame(frame)
                cv2.imshow("实时视频OCR", processed_frame)

            key = cv2.waitKey(1) & 0xFF

            # 按键处理
            if key == ord("q"):
                # 退出程序
                self.is_running = False
            elif key == ord("r") and not selecting_roi:
                # 开始选择ROI
                selecting_roi = True

                # 设置鼠标回调
                def mouse_callback(event, x, y, flags, param):
                    self.mouse_x, self.mouse_y = x, y

                    if event == cv2.EVENT_LBUTTONDOWN:
                        # 左键按下，记录起点
                        nonlocal roi_start_point
                        roi_start_point = (x, y)

                    elif event == cv2.EVENT_LBUTTONUP and roi_start_point:
                        # 左键释放，确定ROI
                        nonlocal selecting_roi
                        x1, y1 = roi_start_point
                        x2, y2 = x, y

                        # 确保坐标正确（左上角和右下角）
                        x_min = min(x1, x2)
                        y_min = min(y1, y2)
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)

                        self.set_roi(x_min, y_min, width, height)

                        # 结束ROI选择模式
                        selecting_roi = False
                        roi_start_point = None

                cv2.setMouseCallback("实时视频OCR", mouse_callback)
                self.mouse_x, self.mouse_y = 0, 0

            elif key == ord("c"):
                # 清除ROI
                self.clear_roi()

        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        """停止OCR系统"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


import threading


def ocr_worker(frame_queue, result_queue):
    """
    OCR工作线程函数

    Args:
        frame_queue: 帧队列，从主线程获取待处理的帧
        result_queue: 结果队列，将OCR结果返回给主线程
    """
    # 创建一个专用的OCR实例
    ocr = PaddleOCR(
        det_model_dir="/Volumes/应用/autotest-system/ch_PP-OCRv3_det_slim_infer",
        rec_model_dir="/Volumes/应用/autotest-system/ch_PP-OCRv3_rec_slim_infer",
        use_angle_cls=False,
        lang="ch",
        use_gpu=True,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_limit_side_len=640,
        rec_batch_num=1,
        # 注意：移除了enable_mkldnn=True选项
        use_mp=False,  # 在线程中不使用多进程
    )

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:  # 退出信号
                break

            try:
                start_time = time.time()  # 记录开始时间
                ocr_result = ocr.ocr(frame, cls=False)
                end_time = time.time()  # 记录结束时间
                single_inference_time = end_time - start_time
                print(
                    f"Single OCR inference time: {single_inference_time:.4f} seconds"
                )  # 打印耗时
                result = (
                    ocr_result[0] if ocr_result and ocr_result[0] is not None else []
                )
                result_queue.put(result)
            except Exception as e:
                print(f"OCR线程处理出错: {e}")
                result_queue.put([])


def main():
    import queue
    import threading  # 再次导入 threading 模块以示清晰

    frame_queue = queue.Queue(maxsize=2)  # 可能需要适当增大队列大小
    result_queue = queue.Queue()

    ocr_system = RealtimeVideoOCR(use_gpu=False, lang="ch")  # 示例继续使用CPU模式

    if ocr_system.start_capture(0, resolution=(1280, 720)):

        # --- 修改部分开始 ---
        num_workers = 32  # 设定你想使用的worker线程数量
        ocr_threads = []
        print(f"启动 {num_workers} 个 OCR 工作线程...")
        for i in range(num_workers):
            # 创建多个线程，都运行 ocr_worker 函数，共享队列
            thread = threading.Thread(
                target=ocr_worker, args=(frame_queue, result_queue), daemon=True
            )
            ocr_threads.append(thread)
            thread.start()
        # --- 修改部分结束 ---

        def run_with_thread():
            try:
                ocr_system.run()
            finally:
                # --- 修改部分开始 ---
                print("发送退出信号并等待 OCR 线程结束...")
                # 向队列发送与 worker 数量相等的 None 信号
                for _ in range(num_workers):
                    frame_queue.put(None)

                # 等待所有 worker 线程结束
                for thread in ocr_threads:
                    thread.join(timeout=2.0)  # 可以设置一个合理的等待超时时间
                print("OCR 线程已停止.")
                # --- 修改部分结束 ---

        original_process_frame = ocr_system.process_frame

        def threaded_process_frame(frame):
            # ... (这部分逻辑保持不变，它负责将帧放入队列和从队列获取结果)
            if ocr_system.roi:
                x, y, w, h = ocr_system.roi
                x = max(0, x)
                y = max(0, y)
                w = min(frame.shape[1] - x, w)
                h = min(frame.shape[0] - y, h)
                process_frame = frame[y : y + h, x : x + w]
            else:
                process_frame = frame

            current_time = time.time()
            if current_time - ocr_system.last_ocr_time >= ocr_system.ocr_interval:
                # 如果队列未满，添加到队列中进行处理
                if not frame_queue.full():  # 检查队列是否满
                    frame_queue.put(process_frame)
                    ocr_system.last_ocr_time = current_time
                    # print("Frame put into queue") # 可以用于调试
                else:
                    print(
                        "Frame queue full, skipping frame"
                    )  # 可以用于调试，观察是否丢帧

            # 检查是否有OCR结果可用
            # 这里使用 while 循环，可以获取到 worker 处理完的所有当前队列中的结果
            while not result_queue.empty():
                try:
                    ocr_system.ocr_results = (
                        result_queue.get_nowait()
                    )  # 使用 get_nowait 避免阻塞
                    # print("Result got from queue") # 可以用于调试
                except queue.Empty:
                    pass  # 队列为空，继续
                except Exception as e:
                    print(f"Error getting result from queue: {e}")

            # 使用原始方法处理显示
            return original_process_frame(frame)

        ocr_system.process_frame = threaded_process_frame

        run_with_thread()


if __name__ == "__main__":
    main()
