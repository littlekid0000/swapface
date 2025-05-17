import os
import base64
import threading
import queue
import numpy as np
import cv2
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from face_swap_insight import swap_face

# 创建 Flask 应用
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# 创建一个安全的队列
image_queue = queue.Queue(maxsize=5)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    try:
        if image_queue.full():
            print("[⚠️] 队列已满，丢弃一帧")
            return
        image_queue.put((data, request.sid))
    except Exception as e:
        print(f"[ERROR] 接收图像失败: {e}")

def image_worker():
    while True:
        try:
            data, sid = image_queue.get()

            # 解码摄像头画面
            header, encoded = data['image_data'].split(",", 1)
            img_bytes = base64.b64decode(encoded)
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            # 解码源人脸图片
            header_face, encoded_face = data['face_image'].split(",", 1)
            src_bytes = base64.b64decode(encoded_face)
            src_image = cv2.imdecode(np.frombuffer(src_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None or src_image is None:
                print("[ERROR] 解码失败，跳过处理")
                continue

            # 执行换脸
            result, total_faces, fps = swap_face(src_image, frame, face_index=0)

            # 编码换脸结果回传前端
            _, buffer = cv2.imencode('.jpg', result)
            b64_result = base64.b64encode(buffer).decode('utf-8')

            socketio.emit('response_back', {
                'image_data': f'data:image/jpeg;base64,{b64_result}',
                'total_faces': total_faces,
                'fps': fps
            }, room=sid)

        except Exception as e:
            print(f"[ERROR] 处理图像任务失败: {e}")

# 启动后台工作线程
threading.Thread(target=image_worker, daemon=True).start()

if __name__ == '__main__':
    # 获取环境变量中的端口，如果没有则使用默认值 5000
    port = int(os.environ.get('PORT', 5000))
    # 获取环境变量中的主机，如果没有则使用默认值 0.0.0.0
    host = os.environ.get('HOST', '0.0.0.0')
    # 获取环境变量中的调试模式，如果没有则使用默认值 False
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
