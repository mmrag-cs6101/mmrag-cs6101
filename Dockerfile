FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV TZ=America/New_York \
    DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y curl git wget tmux nano python3.10 python3.10-dev python3.10-distutils libsm6 libxext6 libgl1-mesa-glx python3-opencv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install uv
RUN uv pip install -r requirements.txt --system
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
COPY . .