# 使用官方 Python 3.10 精简版镜像
FROM python:3.10-slim

# 安装 OpenCV 运行所需的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件
COPY . .

# Fly.io 默认使用 8080 端口
ENV PORT 8080

# 使用 gunicorn 启动 Flask + SocketIO
CMD ["gunicorn", "-b", "0.0.0.0:8080", "wsgi:app"] 