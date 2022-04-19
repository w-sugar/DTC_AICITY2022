Amazing Results With Limited Data In Multi-Class Product Counting and Recegnition.
===

This project (based on [mmdetection](https://github.com/open-mmlab/mmdetection) && [mmclassification](https://github.com/open-mmlab/mmclassification) && [DeepSort](https://github.com/nwojke/deep_sort)) is the re-implementation of our paper.

## Introduction

In light of challenges and the characteristic of Automated retail checkout, we propose a precise and efficient framework. In the training stage, firstly, we use MSRCR to process training data, which has perfect performance in image enhancement by applying it on each color channel independently. Secondly, the processed data can be used to train the classifier, and it can also be randomly pasted into the background to train the detector. In the testing stage, we first preprocess the video, detect the white tray and the human hand area, then detect, track and classify the products in the white tray, and finally process the trajectory through the MTCR algorithm and output the final results.

![introfig](./images/intro.png)

## Data Preparing

### 1. Training
1. Download images and annotations for training detection from [GoogleDrive-det](https://drive.google.com/file/d/1wmKbIXMMPBAw-XhpgbCr6V8Mn8EtRVZh/view?usp=sharing).
2. Download images for training classification from [GoogleDrive-cls](https://drive.google.com/file/d/1k1k6b-cQ9UEh5_L3pVi1DHuYeqovi2Va/view?usp=sharing).
```
data
├── coco
│   └── annotations
│   └── train2017
│   └── val2017
├── cls
│   └── meta
│   └── train
│   └── val
```

### 2. Testing
Please place the videos you want to test in the [test_videos](./test_videos) folder.
```
test_videos/
├── testA_1.mp4
├── testA_2.mp4
├── testA_3.mp4
├── testA_4.mp4
├── testA_5.mp4
├── video_id.txt
```

## Quick & Easy Start

### 1. Environments settings

* python 3.7.12
* pytorch 1.10.0
* torchvision 0.11.1
* cuda 10.2
* mmcv-full 1.4.2
* tensorflow-gpu 1.15.0

```
1. conda create -n DTC python=3.7
2. conda activate DTC
3. git clone https://github.com/w-sugar/DTC_AICITY2022
4. cd DTC_AICITY2022
5. pip install -r requirements.txt
6. python mmdetection/setup.py build develop
7. python mmclassification/setup.py build develop
8. [Data Preparing](##Data Preparing)
```

### 2. Train/Test:

# Contact

If you have any questions, feel free to contact Junfeng Wan (wanjunfeng@bupt.edu.cn).
