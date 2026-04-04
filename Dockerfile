FROM python:3.10-slim

# 设置环境变量，优化 Python 运行和 Pip 缓存
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. 替换 Debian 软件源为国内镜像 (阿里云/华为云均可)
# 针对 Debian 12 (bookworm/trixie) 镜像，优先处理新的 debian.sources 格式
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 2. 安装基础依赖 (OpenCV 运行必需的系统库)
# 特别添加了 libgl1-mesa-glx，防止 OpenCV 报错
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 3. 复制依赖文件并使用国内 Pip 镜像安装
# 火山引擎访问国内 PyPI 镜像速度极快
COPY requirements.txt .
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. 复制项目文件
COPY . .

EXPOSE 5000

# 使用 Gunicorn 启动，适合生产环境
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]