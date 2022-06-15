VERSION=1.1
docker run \
    -it --rm --gpus all \
    --shm-size 4GB \
    --network host \
    andretelfer/hohlbaum-black-mouse-dataset-pytorch:$VERSION