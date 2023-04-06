# Agriculture-Orange-Tree
Exploration of custom orange tree dataset and some applications with CV models


## 1. Setting up environment

`conda create -n yolo python=3.9`

`conda activate yolo`

`pip install ultralytics`

`pip install clearml`

`clearml-init`

Note: Copy & Paste credentials from Setting-> Workspace in you ClearML account

## 2. Train & monitoring

`python train.py`

## 3. Predict

`python predict.py`

## 4. Make the output video lighter 

`ffmpeg -i demo-orange.avi demo-orange-ai.mp4`