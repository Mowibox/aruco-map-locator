FROM ros:jazzy-ros-base

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    build-essential \
    cmake \
    libopencv-dev \
    nano \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN rm /usr/lib/python3.*/EXTERNALLY-MANAGED

COPY requirements-headless.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-headless.txt

WORKDIR /root
RUN git clone https://github.com/Mowibox/aruco-map-locator.git

WORKDIR /root/aruco-map-locator/ros2_ws
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build"

RUN echo "source /root/aruco-map-locator/ros2_ws/install/setup.bash" >> ~/.bashrc
