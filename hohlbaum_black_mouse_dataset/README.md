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

### Test
To test if it's working, simple run
```
docker run --gpus all --shm-size 4GB -it --rm andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest
```
- In order to run this step, you need an Nvidia GPU (tested on 6GB of VRAM)
- This should create a log folder in your current directory with the results in it

### Run scripts
Benchmark script (will take a while: 5 shuffles of 50 epochs)
```
docker run --gpus all --shm-size 4GB -it --rm --network host -v `pwd`/runs:/workspace/runs andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest bash -c scripts/benchmark.sh
```


### Using as an environment
```
docker run --gpus all --shm-size 4GB -it --rm --network host -v `pwd`/runs:/workspace/runs andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest bash 
```

Once in the bash, you can launch jupyter lab and access it from your host machine (the `--network host` option was for this)

Or you could try running the benchmark with custom flags (e.g. more shuffles)
```
python benchmark/train.py --shuffles 3 --epochs 10
```
