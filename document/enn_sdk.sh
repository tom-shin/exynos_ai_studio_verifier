#!/bin/bash
NPUC_DOCKER_NAME="ubuntu-22.04/enntools"
#NPUC_DOCKER_NAME="eht"
DOCKER_BUILD_MODE="rel"
DOCKER_VERSION="7.16.17.21" # 수정
#DOCKER_IMAGE_NAME=${NPUC_DOCKER_NAME}:${DOCKER_VERSION}
DOCKER_IMAGE_NAME=${NPUC_DOCKER_NAME}-${DOCKER_BUILD_MODE}:${DOCKER_VERSION}

if [ $1 ]; then
    CONTAINER_NAME=test-gui-${DOCKER_IMAGE_NAME//[:\/]/-}-$(id -u -n $USER)
else
    CONTAINER_NAME=CPU-${DOCKER_IMAGE_NAME//[:\/]/-}-$(id -u -n $USER)
fi

WORKSPACE=$HOME/npuc-workspace
echo "DOCKER_IMAGE_NAME: ${DOCKER_IMAGE_NAME}"
echo "CONTAINER_NAME: ${CONTAINER_NAME}"

docker ps -a | grep ${CONTAINER_NAME} > /dev/null 2>&1
result=$?

if [ ! $result -eq 0 ]; then
    echo "A docker is running..."
    mkdir -p $WORKSPACE
    if [ "$#" -ne 1 ]; then
        echo "--------------------------------------------------------------------------------------------------------------------------------------"
        echo "Run ENN SDK - CPU ${DOCKER_VERSION} version "
        echo "--------------------------------------------------------------------------------------------------------------------------------------"
        docker run -it\
            --privileged -v /dev/bus/usb:/dev/bus/usb \
            --security-opt seccomp:unconfined \
            --cap-add=ALL --privileged \
            -e http_proxy=$http_proxy \
            -e https_proxy=$https_proxy \
            -e DISPLAY=$DISPLAY \
            -v $HOME/reposit:/home/user/reposit \
            -v $HOME/_project:/home/user/_project \
            -v $HOME/enntools-workspace:/home/user/enntools-workspace \
            -w /home/user/ \
            --restart=always \
            -e LOCAL_USER_ID=`id -u $USER` \
            --name=${CONTAINER_NAME} \
            $DOCKER_IMAGE_NAME /bin/bash
    fi

    if [ $1 ]; then
        echo "--------------------------------------------------------------------------------------------------------------------------------------"
        echo "Run ENN SDK - GPU ${DOCKER_VERSION} version"
        echo "--------------------------------------------------------------------------------------------------------------------------------------"
        docker run -it --gpus all  \
            --privileged -v /dev/bus/usb:/dev/bus/usb \
            --security-opt seccomp:unconfined \
            --cap-add=ALL --privileged \
            --network=bridge \
            --restart=always \
            --pid=host \
            -v $HOME/reposit:/home/user/reposit \
            -v $HOME/_project:/home/user/_project \
            -v $HOME/enntools-workspace:/home/user/enntools-workspace \
            -w /home/user/ \
            --restart=always \
            -e LOCAL_USER_ID=`id -u $USER` \
            --name=${CONTAINER_NAME} \
            $DOCKER_IMAGE_NAME /bin/bash
    fi
else
    echo "A docker is starting..."
    docker start ${CONTAINER_NAME} && docker attach ${CONTAINER_NAME}
fi
