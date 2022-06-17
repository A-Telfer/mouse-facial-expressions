VERSION=1.2
docker run \
    -it --rm --gpus all \
    --shm-size 4GB \
    --network host \
    -v `pwd`:/workspace/logs
    andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest