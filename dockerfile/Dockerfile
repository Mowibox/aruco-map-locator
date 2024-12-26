FROM --platform=linux/arm64 ros:jazzy

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

#To remove the error "setup.py is deprecated" during build
RUN pip install setuptools==58.2.0

RUN pip install transforms3d \
                numpy \
                opencv-python-headless \
                opencv-contrib-python \
                flask \
                pyzbar 

WORKDIR /root
RUN git clone https://github.com/Mowibox/aruco-map-locator.git

WORKDIR /root/aruco-map-locator/ros2_ws
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build"

RUN echo "source /root/aruco-map-locator/ros2_ws/install/setup.bash" >> ~/.bashrc
