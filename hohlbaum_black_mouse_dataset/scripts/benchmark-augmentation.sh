docker run \
    -it \
    --rm \
    --gpus all \
    --shm-size 4GB \
    --network host \
    --volume `pwd`/runs:/workspace/runs \
    andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest bash -c scripts/benchmark-augmentation-docker.sh
