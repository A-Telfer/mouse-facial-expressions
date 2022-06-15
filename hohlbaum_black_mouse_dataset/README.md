# 

## Run Benchmark

Pull the docker image
```
docker pull andretelfer/hohlbaum-black-mouse-dataset-pytorch:1.1
```

Run the benchmark
```
docker run --gpus all --shm-size 4GB -it --rm andretelfer/hohlbaum-black-mouse-dataset-pytorch:1.1
```
- In order to run this step, you need an Nvidia GPU (tested on 6GB of VRAM)