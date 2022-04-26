# Build Docker image to run code for ml-project

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
LABEL org.opencontainers.image.authors="ddp5730@rit.edu"

RUN apt update && apt install -y build-essential python3 python3-pip git
RUN apt clean

RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install cupy
RUN pip3 install scipy
RUN pip3 install tqdm
RUN pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install git+https://github.com/nottombrown/imagenet_stubs

RUN useradd --create-home --shell /bin/bash poppfd  # Update this to your desired user

RUN pip3 install timm
RUN pip3 install tensorboard
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install imblearn