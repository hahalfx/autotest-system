"""
OCR模块测试程序
验证EasyOCR的基本功能
"""
import logging
import cv2
from src.vision.camera.processor import VideoProcessor
from src.vision.config.config_model import CameraConfig, AnalysisConfig
import numpy as np
from src.vision.analysis.ocr_processor import OCRProcessor, OCRConfig

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认配置
config = OCRConfig(
    languages=["ch_sim", "en"],
    gpu=True,
    confidence_threshold=0.1,  # 设置较低的置信度阈值以便测试
    model_storage="./models/ocr"
)

def process_video_stream():
    """
    处理视频流并进行OCR检测
    """
    try:
        # 配置摄像头参数
        camera_config = CameraConfig(
            device_id=0,  # 默认摄像头
            resolution=(480, 270),
            fps=30
        )
        
        # 配置分析参数
        analysis_config = AnalysisConfig(
            frame_skip=2,  # 跳过帧数以提高性能
            #resize_dimension=(480, 270),  # 保持较小分辨率
            roi=(0, 0, 200, 200),  # 限制OCR处理区域
        )
                
        # 创建视频处理器
        video_processor = VideoProcessor(camera_config, analysis_config)
        
        # 创建OCR处理器
        ocr_processor = OCRProcessor(config=config)
        
        # 初始化OCR处理器
        if not ocr_processor.initialize():
            logger.error("OCR处理器初始化失败")
            return
        
        # 启动视频处理
        video_processor.start()
        logger.info("视频处理启动")
        
        try:
            while True:
                # 获取最新帧
                frame = video_processor.get_latest_frame()
                
                if frame is not None:
                    # 执行OCR分析
                    result = ocr_processor.analyze(frame)
                    
                    # 显示结果
                    if result.success:
                        logger.info(f"OCR识别成功:")
                        logger.info(f"文本: {result.data['text']}")
                        logger.info(f"置信度: {result.data['confidence']:.2f}")
                        logger.info(f"检测到的文本框: {len(result.data['boxes'])}")
                        
                        # 验证是否包含"测评"二字
                        has_keyword = "测评" in result.data['text']
                        status_text = "验证成功: 找到'测评'" if has_keyword else "验证失败: 未找到'测评'"
                        status_color = (0, 255, 0) if has_keyword else (0, 0, 255)
                        
                        # 在画面顶部显示验证结果
                        cv2.putText(frame, status_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                        
                        # 在画面上绘制结果
                        for points in result.data['boxes']:
                            # 将点列表转换为轮廓格式
                            contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                            
                            # 绘制多边形
                            cv2.polylines(frame, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
                            
                            # 显示置信度
                            first_point = points[0]
                            position = (int(first_point[0]), int(first_point[1])-10)
                            cv2.putText(frame, 
                                        f"{result.data['confidence']:.2f}", 
                                        position,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                        (0, 255, 0), 2)
                    else:
                        logger.error(f"OCR识别失败: {result.error}")
                
                # 显示画面
                if frame is not None:
                    cv2.imshow("OCR Video Stream", frame)
                
                # 按下 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # 停止视频处理
            video_processor.stop()
            cv2.destroyAllWindows()
            logger.info("视频处理停止")
        
    except Exception as e:
        logger.error(f"视频OCR测试异常: {str(e)}")


if __name__ == "__main__":
    logger.info("开始视频OCR模块测试...")
    process_video_stream()
    logger.info("视频OCR模块测试完成")
