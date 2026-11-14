
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=0

RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-numpy \
    python3-scipy \
    python3-pytest \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ros2_ws

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc

CMD ["bash"]
