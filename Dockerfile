# Build Docker image specific to SHOT

FROM ml-base

RUN pip3 install timm
RUN pip3 install tensorboard
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install imblearn