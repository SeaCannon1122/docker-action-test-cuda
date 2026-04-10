FROM ultralytics/ultralytics:latest-jetson-jetpack6

SHELL ["/bin/bash", "-c"]

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
