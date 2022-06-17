mkdir runs
docker run \
    -it --rm --gpus all \
    --shm-size 4GB \
    --network host \
    -v `pwd`:/workspace/shared \
    andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest \
    bash 