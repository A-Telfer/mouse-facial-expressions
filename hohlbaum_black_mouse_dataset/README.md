# Benchmark of BMv1 Dataset
Unofficial implementation of detecting mouse well-being using facial expressions in PyTorch 

Paper
> Andresen, N., Wöllhaf, M., Hohlbaum, K., Lewejohann, L., Hellwich, O., Thöne-Reineke, C., & Belik, V. (2020). Towards a fully automated surveillance of well-being status in laboratory mice using deep learning: Starting with facial expression analysis. PLoS One, 15(4), e0228059.

Dataset
> Hohlbaum, K., Andresen, N., Wöllhaf, M., Lewejohann, L., Hellwich, O., Thöne-Reineke, C., & Belik, V. (2019). Black Mice Dataset v1.

## Docker usage

### Downloading the image
Pull the docker image
```
docker pull andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest
```

### Run Benchmark
Run the benchmark
```
docker run --gpus all --shm-size 4GB -it --rm andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest
```
- In order to run this step, you need an Nvidia GPU (tested on 6GB of VRAM)

### Using as an environment
```
docker run --gpus all --shm-size 4GB -it --rm --network host andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest bash 
```

Once in the bash, you can launch jupyter lab and access it from your host machine
