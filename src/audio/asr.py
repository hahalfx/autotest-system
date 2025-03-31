import pyaudio
from dashscope.audio.asr import (Recognition, RecognitionCallback,
                                 RecognitionResult)
import dashscope
import os
from dotenv import load_dotenv

load_dotenv()


class ASRModule:
    def __init__(self):
        self.mic = None
        self.stream = None
        self.recognition = None

    def start(self, result_callback=None):
        """启动语音识别
        Args:
            result_callback: 识别结果回调函数
        """
        print("INFO: ASR module start method called")
        
        # 如果已经在运行，先停止
        if self.recognition:
            print("INFO: ASR module already running, stopping first")
            self.stop()
            
        try:
            # 获取API密钥
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY environment variable not set")
                
            dashscope.api_key = api_key
            print("INFO: Creating ASR callback")
            callback = self.Callback(self, result_callback)
            
            print("INFO: Initializing recognition service")
            self.recognition = Recognition(
                model='paraformer-realtime-v2',
                format='pcm',
                sample_rate=16000,
                callback=callback
            )
            
            print("INFO: Starting recognition service")
            self.recognition.start()
            print("INFO: Recognition service started successfully")
        except Exception as e:
            print(f"ERROR: Failed to start ASR module: {e}")
            # Clean up if initialization fails
            if self.recognition:
                try:
                    self.recognition.stop()
                except:
                    pass
                self.recognition = None

    def stop(self):
        """停止语音识别"""
        print("INFO: ASR module stop method called")
        
        # 先停止识别服务
        if self.recognition:
            try:
                print("INFO: Stopping recognition service")
                self.recognition.stop()
            except Exception as e:
                print(f"ERROR: 停止识别时出错: {e}")
            finally:
                self.recognition = None
            
        # 然后关闭音频流
        if self.stream:
            try:
                print("INFO: Stopping and closing audio stream")
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"ERROR: 关闭音频流时出错: {e}")
            finally:
                self.stream = None
            
        # 最后终止音频设备
        if self.mic:
            try:
                print("INFO: Terminating audio device")
                self.mic.terminate()
            except Exception as e:
                print(f"ERROR: 终止音频设备时出错: {e}")
            finally:
                self.mic = None
                
        print("INFO: ASR module stopped completely")

    class Callback(RecognitionCallback):
        def __init__(self, parent, result_callback=None):
            self.parent = parent
            self.result_callback = result_callback

        def on_open(self) -> None:
            print('INFO: ASR recognition service opened (on_open callback)')
            try:
                print('INFO: Initializing audio device')
                self.parent.mic = pyaudio.PyAudio()
                
                print('INFO: Opening audio stream')
                self.parent.stream = self.parent.mic.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True
                )
                print('INFO: Audio stream opened successfully')
                print('ASR模块已启动，请开始说话...')
            except Exception as e:
                print(f'ERROR: Error initializing audio in on_open: {e}')
                # Try to clean up if initialization fails
                if self.parent.stream:
                    try:
                        self.parent.stream.close()
                    except:
                        pass
                    self.parent.stream = None
                    
                if self.parent.mic:
                    try:
                        self.parent.mic.terminate()
                    except:
                        pass
                    self.parent.mic = None

        def on_close(self) -> None:
            print('\nINFO: ASR recognition service closed (on_close callback)')
            # Clean up resources in the callback
            if self.parent.stream:
                try:
                    print('INFO: Stopping and closing audio stream in on_close callback')
                    self.parent.stream.stop_stream()
                    self.parent.stream.close()
                except Exception as e:
                    print(f'ERROR: 关闭音频流时出错 (on_close): {e}')
                self.parent.stream = None
                
            if self.parent.mic:
                try:
                    print('INFO: Terminating audio device in on_close callback')
                    self.parent.mic.terminate()
                except Exception as e:
                    print(f'ERROR: 终止音频设备时出错 (on_close): {e}')
                self.parent.mic = None
                
            print('INFO: ASR on_close callback completed')

        def on_event(self, result: RecognitionResult) -> None:
            # Check if parent recognition is still active
            if not self.parent.recognition:
                print('INFO: Received ASR event but recognition is no longer active, ignoring')
                return
                
            try:
                sentence = result.get_sentence()
                print('识别结果:', sentence)
                
                if self.result_callback:
                    # Only call the callback if parent is still active
                    if self.parent.recognition:
                        self.result_callback(sentence)
                    else:
                        print('INFO: Not calling result_callback as recognition is no longer active')
            except Exception as e:
                print(f'ERROR: Error processing ASR event: {e}')

if __name__ == "__main__":
    import signal
    import sys

    asr = ASRModule()
    
    def signal_handler(sig, frame):
        asr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    print("启动语音识别模块，按Ctrl+C退出...")
    asr.start()
    
    try:
        while True:
            if asr.stream:
                data = asr.stream.read(3200, exception_on_overflow=False)
                asr.recognition.send_audio_frame(data)
    except KeyboardInterrupt:
        asr.stop()
