#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import websockets
import json
import cv2
import numpy as np
import time
import base64
from paddleocr import PaddleOCR
import threading
import queue
from typing import Dict, Any, List, Tuple, Optional

# OCR worker 线程函数
def ocr_worker(frame_queue, result_queue, settings):
    """
    OCR工作线程函数
    
    Args:
        frame_queue: 帧队列，从主线程获取待处理的帧
        result_queue: 结果队列，将OCR结果返回给主线程
        settings: OCR设置参数
    """
    # 创建一个专用的OCR实例
    ocr = PaddleOCR(
        det_model_dir=settings.get("det_model_dir", None),
        rec_model_dir=settings.get("rec_model_dir", None),
        use_angle_cls=False,
        lang=settings.get("lang", "ch"),
        use_gpu=settings.get("use_gpu", False),
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_limit_side_len=640,
        rec_batch_num=1,
        use_mp=False,  # 在线程中不使用多进程
    )
    
    print(f"OCR worker started with settings: {settings}")
    
    while True:
        if not frame_queue.empty():
            frame_data = frame_queue.get()
            if frame_data is None:  # 退出信号
                break
                
            frame, frame_id, meta_data = frame_data

            try:
                # 验证图像数据
                if frame is None or frame.size == 0:
                    raise ValueError("Invalid frame data")
                
                start_time = time.time()  # 记录开始时间
                ocr_result = ocr.ocr(frame, cls=False)
                end_time = time.time()  # 记录结束时间
                inference_time = end_time - start_time
                
                # 处理OCR结果
                result_list = []
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        box_points = line[0]
                        text = line[1][0]
                        confidence = float(line[1][1])

                        print(f"OCR结果: {text}, 置信度: {confidence}")
                        
                        # 如果是ROI区域且提供了原始坐标，转换坐标到原始图像坐标系
                        if meta_data.get("is_roi") and meta_data.get("roi_coords"):
                            roi_x, roi_y = meta_data["roi_coords"][0], meta_data["roi_coords"][1]
                            # 调整坐标点到原始图像中的位置
                            adjusted_box = []
                            for point in box_points:
                                adjusted_box.append([point[0] + roi_x, point[1] + roi_y])
                            box_points = adjusted_box
                        
                        result_list.append({
                            "box": box_points,
                            "text": text,
                            "confidence": confidence
                        })
                
                result_queue.put({
                    "frame_id": frame_id,
                    "results": result_list,
                    "inference_time": inference_time,
                    "meta_data": meta_data
                })
                
            except Exception as e:
                error_msg = f"OCR线程处理出错: {str(e)}. Frame shape: {frame.shape if frame is not None else 'None'}"
                print(error_msg)
                result_queue.put({
                    "frame_id": frame_id,
                    "results": [],
                    "error": error_msg,
                    "meta_data": meta_data
                })

class OCRServer:
    def __init__(self):
        self.clients = set()
        self.frame_queue = queue.Queue(maxsize=10)  # 增加队列大小，适应前端控制的发送频率
        self.result_queue = queue.Queue()
        self.ocr_workers = []
        self.num_workers = 16  # 默认工作线程数
        self.ocr_settings = {
            "lang": "ch",
            "use_gpu": False,
            "det_model_dir": "/Volumes/应用/autotest-system/ch_PP-OCRv3_det_slim_infer",
            "rec_model_dir": "/Volumes/应用/autotest-system/ch_PP-OCRv3_rec_slim_infer"
        }
        self.roi = None
        self.ocr_interval = 0.5  # 默认OCR处理间隔，现在仅作为初始设置返回给前端
        self.active_connections = {}  # 存储活跃的客户端连接
        self.next_frame_id = 0  # 帧ID计数器

    def start_ocr_workers(self):
        """启动OCR工作线程"""
        print(f"启动 {self.num_workers} 个 OCR 工作线程...")
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=ocr_worker, 
                args=(self.frame_queue, self.result_queue, self.ocr_settings), 
                daemon=True
            )
            self.ocr_workers.append(thread)
            thread.start()
    
    def stop_ocr_workers(self):
        """停止OCR工作线程"""
        print("发送退出信号并等待 OCR 线程结束...")
        for _ in range(self.num_workers):
            self.frame_queue.put(None)
        
        for thread in self.ocr_workers:
            thread.join(timeout=2.0)
        print("OCR 线程已停止")
        
        self.ocr_workers = []

    async def register(self, websocket):
        """注册新的WebSocket客户端连接"""
        client_id = id(websocket)
        self.clients.add(websocket)
        self.active_connections[client_id] = {
            "websocket": websocket,
            "last_frame_time": time.time()
        }
        return client_id

    async def unregister(self, websocket):
        """注销WebSocket客户端连接"""
        client_id = id(websocket)
        self.clients.remove(websocket)
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def process_message(self, websocket, message):
        """处理从客户端接收的消息"""
        try:
            if isinstance(message, bytes):
                # 处理二进制帧消息(包含元数据+图像数据)
                try:
                    # 尝试解析消息开头的JSON元数据
                    try:
                        # JSON元数据应该位于消息开头
                        decoded = message.decode('utf-8')
                        json_end = decoded.find('}')
                        if json_end == -1:
                            raise ValueError("Invalid metadata format")
                            
                        meta_data = json.loads(decoded[:json_end+1])
                        image_data = message[len(decoded[:json_end+1].encode('utf-8')):]
                    except UnicodeDecodeError:
                        # 如果UTF-8解码失败，尝试从开头查找JSON特征
                        try:
                            json_start = message.find(b'{')
                            json_end = message.find(b'}')
                            if json_start == -1 or json_end == -1:
                                raise ValueError("Cannot locate JSON metadata")
                                
                            meta_data = json.loads(message[json_start:json_end+1].decode('utf-8'))
                            image_data = message[json_end+1:]
                        except Exception as e:
                            raise ValueError(f"Cannot parse metadata: {str(e)}")
                    
                    await self.handle_frame(websocket, {
                        "type": "frame",
                        "frame": image_data,
                        "frame_id": meta_data.get("frame_id", self.next_frame_id),
                        "width": meta_data.get("width"),
                        "height": meta_data.get("height"),
                        "is_roi": meta_data.get("is_roi", False),
                        "original_width": meta_data.get("original_width"),
                        "original_height": meta_data.get("original_height"),
                        "roi_coords": meta_data.get("roi_coords")
                    })
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Error processing binary message: {str(e)}"
                    }))
            else:
                # 处理文本配置消息
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "frame":
                    # 处理文本格式的视频帧(兼容旧版)
                    await self.handle_frame(websocket, data)
                elif message_type == "config":
                    # 处理配置消息
                    await self.handle_config(websocket, data)
                elif message_type == "ping":
                    # 处理心跳消息
                    await websocket.send(json.dumps({"type": "pong"}))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }))
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON message"
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Error processing message: {str(e)}"
            }))

    async def handle_frame(self, websocket, data):
        """处理接收到的视频帧"""
        client_id = id(websocket)
        current_time = time.time()
        
        # 更新最后一次接收帧的时间
        if client_id in self.active_connections:
            self.active_connections[client_id]["last_frame_time"] = current_time
        
        # 接收二进制数据包
        frame_blob = data.get("frame")
        frame_id = data.get("frame_id", self.next_frame_id)
        self.next_frame_id += 1
        
        # 提取元数据
        meta_data = {
            "is_roi": data.get("is_roi", False),
            "original_width": data.get("original_width"),
            "original_height": data.get("original_height"),
            "roi_coords": data.get("roi_coords"),
            "width": data.get("width"),
            "height": data.get("height")
        }
        
        try:
            # 处理二进制图像数据
            if not isinstance(frame_blob, (bytes, bytearray)):
                raise ValueError("Frame data must be bytes or bytearray")
            
            nparr = np.frombuffer(frame_blob, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image data")
            
            print(f"Received frame {frame_id}, shape: {frame.shape}, is_roi: {meta_data['is_roi']}")
            
            # 前端已经处理了ROI裁剪，这里直接处理收到的图像
            # 不再需要服务器端控制OCR处理频率，由前端控制发送频率
            if not self.frame_queue.full():
                self.frame_queue.put((frame, frame_id, meta_data))
            else:
                print("Frame queue full, skipping frame")
            
            # 发送确认消息
            await websocket.send(json.dumps({
                "type": "frame_received",
                "frame_id": frame_id
            }))
            
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Error processing frame: {str(e)}"
            }))

    async def handle_config(self, websocket, data):
        """处理配置更新"""
        config = data.get("config", {})
        
        # 更新ROI设置，但不再需要控制OCR间隔（由前端控制）
        if "roi" in config:
            self.roi = config["roi"]
            print(f"ROI updated: {self.roi}")
        
        # 更新OCR设置
        ocr_config = config.get("ocr", {})
        if ocr_config:
            restart_workers = False
            
            for key in ["lang", "use_gpu", "det_model_dir", "rec_model_dir"]:
                if key in ocr_config:
                    old_value = self.ocr_settings.get(key)
                    new_value = ocr_config[key]
                    if old_value != new_value:
                        self.ocr_settings[key] = new_value
                        restart_workers = True
            
            # 更新工作线程数
            if "num_workers" in ocr_config:
                new_workers = max(1, int(ocr_config["num_workers"]))
                if new_workers != self.num_workers:
                    self.num_workers = new_workers
                    restart_workers = True
            
            # 如果关键设置变更，重启工作线程
            if restart_workers and self.ocr_workers:
                self.stop_ocr_workers()
                self.start_ocr_workers()
        
        await websocket.send(json.dumps({
            "type": "config_updated",
            "config": {
                "roi": self.roi,
                "ocr_interval": self.ocr_interval,  # 保留这个值，仅作为参考
                "ocr_settings": self.ocr_settings,
                "num_workers": self.num_workers
            }
        }))

    async def send_results(self):
        """将OCR结果发送给客户端"""
        while True:
            try:
                if not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    
                    # 转换结果为可JSON序列化的格式
                    for item in result.get("results", []):
                        if "box" in item:
                            item["box"] = item["box"].tolist() if isinstance(item["box"], np.ndarray) else item["box"]
                    
                    # 发送给所有连接的客户端
                    if self.clients:
                        message = json.dumps({
                            "type": "ocr_result",
                            "data": result
                        })
                        await asyncio.gather(
                            *[client.send(message) for client in self.clients],
                            return_exceptions=True
                        )
                
                # 短暂暂停，避免CPU占用过高
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error in send_results: {e}")
                await asyncio.sleep(0.1)

    async def handler(self, websocket):
        """处理WebSocket连接"""
        try:
            # 注册新客户端
            client_id = await self.register(websocket)
            print(f"Client connected: {client_id}")
            
            # 发送初始配置
            await websocket.send(json.dumps({
                "type": "init",
                "config": {
                    "roi": self.roi,
                    "ocr_interval": self.ocr_interval,  # 仅作为初始参考值
                    "ocr_settings": self.ocr_settings,
                    "num_workers": self.num_workers
                }
            }))
            
            # 处理消息
            async for message in websocket:
                await self.process_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed")
        finally:
            # 注销客户端
            await self.unregister(websocket)

    async def monitor_connections(self):
        """监控连接状态，处理超时的连接"""
        while True:
            current_time = time.time()
            to_remove = []
            
            for client_id, info in self.active_connections.items():
                last_time = info["last_frame_time"]
                websocket = info["websocket"]
                
                # 如果30秒没有收到消息，认为连接已断开
                if current_time - last_time > 30:
                    to_remove.append((client_id, websocket))
            
            # 关闭超时的连接
            for client_id, websocket in to_remove:
                try:
                    await websocket.close()
                    print(f"Closed inactive connection: {client_id}")
                except:
                    pass
                
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                if websocket in self.clients:
                    self.clients.remove(websocket)
            
            await asyncio.sleep(5)  # 每5秒检查一次

    async def serve(self, host="0.0.0.0", port=8765):
        """启动WebSocket服务器"""
        # 启动OCR工作线程
        self.start_ocr_workers()
        
        # 创建WebSocket服务器
        server = await websockets.serve(self.handler, host, port)
        
        # 启动结果发送任务
        results_task = asyncio.create_task(self.send_results())
        
        # 启动连接监控任务
        monitor_task = asyncio.create_task(self.monitor_connections())
        
        print(f"OCR WebSocket Server started on {host}:{port}")
        
        try:
            await server.wait_closed()
        finally:
            # 清理资源
            results_task.cancel()
            monitor_task.cancel()
            self.stop_ocr_workers()

if __name__ == "__main__":
    server = OCRServer()
    
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("Server stopped by user")