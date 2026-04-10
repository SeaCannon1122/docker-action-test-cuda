FROM ultralytics/ultralytics:latest-jetson-jetpack6

SHELL ["/bin/bash", "-c"]

# librealsense

WORKDIR /workspace
RUN apt update && apt-get install -y \
    build-essential cmake git pkg-config \
    libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev libudev-dev libv4l-dev \
    libusb-1.0-0-dev libssl-dev freeglut3-dev mesa-utils mesa-common-dev

RUN git clone https://github.com/realsenseai/librealsense.git
WORKDIR /workspace/librealsense/build
RUN cmake ..
RUN cmake --build . --config Release
RUN cmake --install .

# install some libs

RUN apt install -y unzip git
RUN apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libgstrtspserver-1.0-dev \
    libyaml-cpp-dev \
    libssl-dev \
    wget \
    build-essential \
    pkg-config

# quac

WORKDIR /workspace

COPY . /workspace

WORKDIR /workspace/build

RUN cmake ..
RUN make
