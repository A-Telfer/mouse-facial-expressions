FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update 

# Download the BMv1 dataset for post-surgical pain in mice
# Hohlbaum, K., Andresen, N., Wöllhaf, M., Lewejohann, L., Hellwich, O., Thöne-Reineke, C., & Belik, V. (2019). Black Mice Dataset v1.
RUN apt-get install -y curl
RUN curl https://depositonce.tu-berlin.de/bitstream/11303/10451/2/BMv1.zip -o BMv1.zip

RUN apt-get install -y unzip
RUN unzip BMv1.zip 

# Copy the code over
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Create a pickled version of the dataset
RUN python mouse_facial_expressions/dataset.py BMv1/

