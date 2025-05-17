import time
import cv2
import numpy as np
import os
import sys
import insightface


# 添加 insightface 到 Python 路径
insightface_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insightface')
if os.path.exists(insightface_path):
    sys.path.append(insightface_path)

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: Could not import FaceAnalysis. Please run install_deps.py first.")
    sys.exit(1)

import onnxruntime as ort

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'inswapper_128.onnx')

# 检查模型文件是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}")

# ONNX Runtime配置
so = ort.SessionOptions()
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.add_session_config_entry("session.use_env_allocators", "1")

try:
    # 加载inswapper模型
    swapper = insightface.model_zoo.get_model(
        model_path,
        session_options=so,
        providers=['CPUExecutionProvider']
    )

    # 初始化人脸检测器
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
except Exception as e:
    print(f"初始化模型失败: {e}")
    raise

# 缓存
src_face_cache = {}
last_fps_time = time.time()
fps_counter = 0
fps_value = 0

def resize_to_match_height(img, target_height):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))

def get_src_face(src_img):
    img_hash = hash(src_img.tobytes())
    if img_hash in src_face_cache:
        return src_face_cache[img_hash]

    # 放大到更大尺寸后检测
    resized_src = resize_to_match_height(src_img, 640)  # ❗改大
    src_faces = face_analyzer.get(resized_src)
    if not src_faces:
        print("[⚠️] 源图检测不到人脸")
        return None

    src_face = src_faces[0]
    src_face_cache[img_hash] = src_face
    print(f"[✅] 缓存源人脸成功，位置：{src_face.bbox}")
    return src_face

def swap_face(src_img, dst_img, face_index=0):
    global last_fps_time, fps_counter, fps_value

    # 检测目标人脸
    dst_faces = face_analyzer.get(dst_img)
    if not dst_faces:
        print("[⚠️] 目标图未检测到人脸")
        return dst_img, 0, fps_value

    src_face = get_src_face(src_img)
    if src_face is None:
        print("[⚠️] 无法获取源人脸")
        return dst_img, len(dst_faces), fps_value

    if face_index >= len(dst_faces):
        face_index = 0
    dst_face = dst_faces[face_index]

    try:
        result = swapper.get(dst_img, dst_face, src_face)
    except Exception as e:
        print(f"[ERROR] 换脸出错: {e}")
        return dst_img, len(dst_faces), fps_value

    fps_counter += 1
    now = time.time()
    if now - last_fps_time >= 1.0:
        fps_value = fps_counter
        fps_counter = 0
        last_fps_time = now

    return result, len(dst_faces), fps_value
