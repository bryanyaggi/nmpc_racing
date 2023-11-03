FROM osrf/ros:noetic-desktop-full

# Args for User
ARG UNAME=ubuntu
ARG UID=1000
ARG GID=1000

ENV DISTRO="focal"
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV TERM="xterm-256color"

# Install utilities
RUN apt-get update && apt-get install -y \
        curl \
        iproute2 \
        iputils-ping \
        net-tools \
        python3-pip \
        vim \
        wget \
     && rm -rf /var/lib/apt/lists/*

# Setup repositories and public keys
## ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        libgoogle-glog-dev \
        protobuf-compiler \
        python3-catkin-tools \
        python3-rosdep \
        python3-vcstool \
        python3-wstool \
        qt5-default

# Install ROS packages
RUN apt-get install -y \
        ros-$ROS_DISTRO-joy \
        ros-$ROS_DISTRO-control-toolbox \
        ros-$ROS_DISTRO-navigation \
        ros-$ROS_DISTRO-teb-local-planner* \
        ros-$ROS_DISTRO-ros-control \
        ros-$ROS_DISTRO-ros-controllers \
        ros-$ROS_DISTRO-gazebo-ros-control \
        ros-$ROS_DISTRO-ackermann-msgs \
        ros-$ROS_DISTRO-serial
RUN rosdep update

# Create user
RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# Allow the user to run sudo without a password
RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Login as user
USER $UNAME
WORKDIR ${HOME}

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="${HOME}/.cargo/bin:${PATH}"
RUN pip3 install opengen

# Set default command
CMD ["/bin/bash"]
