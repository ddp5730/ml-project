# MLP for IDS Transfer Learning

This code was created for a project for CMPE-789 Machine Learning for Cybersecurity
Analytics.  The code allows for testing of transfer learning on an 
intrusion detection dataset.  For more details see the included presentation
with an overview and results.

[Presentation](Investigation_of_IDS_Transfer_Learning_with_MLP_Networks.pdf)

---
## Procedure

### Download Dataset
Download the CIC-IDS-2017 and/or the CIC-IDS-2018 dataset to use this code.

CIC-IDS-2018: [link](https://www.unb.ca/cic/datasets/ids-2018.html)

CIC-IDS-2017: [link](https://www.unb.ca/cic/datasets/ids-2017.html)


The code could be adapted for other datasets, but this requires further work.

### Build Docker Container (Optional)
You may use the provided Dockerfile to build a container with all
of the necessary requirements required to run the provided code.
However, you must have some version of CUDA, Docker and the
NVIDIA container toolkit installed (see [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

Otherwise, feel free to set up the environment in whatever way you want.

1. Copy Dockerfile template
   ```
   $ cp Dockerfile Dockerfile.new
   ```
2. Update lines 4 and 18 with desired author info and username
3. Update .dockerignore with any added directories as necessary
4. Build the Docker container
    ```
   $ docker build -t <desired-tag> -f Dockerfile.new .
    ```
5. Run Docker Container.  `<tag>` must be the same tag used in step 4.
   ```
   docker run -it --gpus all --shm-size=25G -e HOME=$HOME -e USER=$USER -v $HOME:$HOME -w $HOME --user <created-user> <tag>
   ```
6. Navigate to code (Home directories will be linked) and run

### Run Random Forest Classifier
The random forest classifier is used as a baseline for the MLP model.  The pkl-path
argument can point to any empty directory where the preprocessed dataset will
be saved to reduce preprocessing time on subsequent runs.

```
python3 classify_rf.py \
--max-depth=10 \
--data-path=/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/ \
--dset='cic-2018' \
--pkl-path=/home/poppfd/College/ML_Cyber/ml-project/data
```

### Run MLP Classifier

There are two files for the MLP classifier.  A training script `mlp.py` and a 
testing script `eval_mlp.py`.  Provided here are the sample commands to run
these scripts.  These commands will have to be updated
to match your environment.

#### Train on 2018 data
```
python3 mlp.py \
--dset=cic-2018 \
--data-root=/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/ \
--pkl-path=/home/poppfd/College/ML_Cyber/ml-project/data \
--batch-size=32 \
--eval-batch-size=1028 \
--num-epochs=10 \
--warmup-epochs=2 \
--learning-rate=1e-4 \
--min-lr=1e-6 \
--warmup-lr=1e-5 \
--name=train-2018-1
```

#### Train 2018 pretrained on 2017 data

Note that the script is identical to freeze the feature extraction layer.
Update `--transfer-learn=freeze-feature`

```
python3 mlp.py \
--dset=cic-2017 \
--data-root=/home/poppfd/data/CIC-IDS2017/MachineLearningCVE \
--pkl-path=/home/poppfd/College/ML_Cyber/ml-project/data \
--batch-size=32 \
--eval-batch-size=1028 \
--num-epochs=10 \
--warmup-epochs=2 \
--learning-rate=1e-4 \
--min-lr=1e-6 \
--warmup-lr=1e-5 \
--transfer-learn=fine-tune \
--source-classes=10 \
--pretrained-path=/home/poppfd/College/ML_Cyber/ml-project/output/mlp-5-layer-3/model_eval_5.pth \
--name=transfer-2017-fine-tune
```

#### Evaluate MLP

```
python3 eval_mlp.py \
--dset='cic-2018' \
--dataset-dir=/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/ \
--pkl-path=/home/poppfd/College/ML_Cyber/ml-project/data \
--model-path=/home/poppfd/College/ML_Cyber/ml-project/output/transfer-2018-freeze-1/model_eval_23.pth \
--batch-size=1028 \
--name=transfer-2018-freeze-1
```

#### Generate TSNE Plots
The `eval_mlp.py` script can also be used to generate t-SNE visualizations of
the MLP feature embedding.  For this case since the t-SNE is really slow for a
high number of samples, only a small subset of the evaluation dataset is used
for the t-SNE plots.  Therefore, all other output of this run should be ignored
as it does not include the full evaluation dataset.

```
python3 eval_mlp.py \
--dset='cic-2018' \
--dataset-dir=/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/ \
--pkl-path=/home/poppfd/College/ML_Cyber/ml-project/data \
--model-path=/home/poppfd/College/ML_Cyber/ml-project/output/transfer-2018-freeze-1/model_eval_23.pth \
--batch-size=1028 \
--tnse \
--tsne-percent=0.01 \
--name=transfer-2018-freeze-1
```



