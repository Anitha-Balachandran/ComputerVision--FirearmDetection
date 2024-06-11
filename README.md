# ComputerVision--FirearmDetection
A deep learning-based project for real-time firearm detection in video surveillance using Faster R-CNN and YOLOv5 models with SRGAN-enhanced data augmentation.

# Enhancing Firearm Detection in Video Surveillance

## Abstract

This project aims to develop a robust, automated firearm detection system using deep learning to enhance real-time surveillance and improve public safety. We analyzed 398 videos from the Mendeley Data set using Faster R-CNN with ResNet-50 and YOLOv5 models. Data augmentation with SRGAN significantly boosted detection accuracy and speed. Faster R-CNN achieved an mAP of 0.995, and YOLOv5 achieved an mAP of 0.983. These results highlight the effectiveness of combining high-accuracy models with fast models, enhanced by SRGAN, for real-time firearm detection.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Dataset Collection](#dataset-collection)
  - [Data Pre-processing](#data-pre-processing)
  - [Modeling](#modeling)
- [Results and Analysis](#results-and-analysis)
- [Conclusion](#conclusion)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors and Acknowledgments](#authors-and-acknowledgments)

## Introduction

The increase in concealed firearm incidents underscores the need for improved security measures. This study aims to leverage deep learning to create a robust, automated firearm detection system. We employed Faster R-CNN with ResNet-50 and YOLOv5 models, integrating SRGAN for data augmentation to enhance detection accuracy and speed.

## Methodology

### Dataset Collection

We used a dataset of 398 videos featuring individuals with and without firearms. The dataset includes categories for Handgun, Machine Gun, and No Gun.
- Source: [Mendeley Data](https://data.mendeley.com/datasets/bbzpxhd22j/2)
- ![Dataset Image](https://github.com/sowmyakuruba20/ComputerVision-FirearmDetection/assets/131414180/6b332077-f585-4172-99ef-447671f4d010)

### Data Pre-processing

Preprocessing involved:
- Extracting frames at 25 FPS and 640x480 resolution
- Converting annotations to the required formats for YOLOv5 and Faster R-CNN
- Applying SRGAN for image enhancement

### Modeling

- **Faster R-CNN with ResNet-50**: This model includes a Region Proposal Network (RPN) and a Fast R-CNN detector. We modified the model to detect Machine Gun and Handgun.
- **YOLOv5**: The model's architecture includes Backbone, Neck, and Head, designed for fast and accurate object detection. We tailored it to detect three specific classes: Machine Gun, Handgun, and No Gun.

## Results and Analysis

- **Faster R-CNN**: Achieved an mAP@0.5 of 99.5% and mAP@0.5:0.95 of 76.7% with GAN-augmented data.
- **YOLOv5**: Achieved an mAP@0.5 of 0.983 and mAP@0.5:0.95 of 0.648 with GAN-augmented data.

### Comparison:
- **Faster R-CNN**: High accuracy but slower inference time.
- **YOLOv5**: Faster inference but slightly lower accuracy.

## Conclusion

This project demonstrates the effectiveness of integrating high-accuracy and fast models with SRGAN for data augmentation in real-time firearm detection. Future research could focus on integrating real-time alert mechanisms and adaptive learning techniques to improve system performance and reliability.

## Demo

### Machine Gun Detection

![Machine Gun Detection](https://github.com/Anitha-Balachandran/ComputerVision--FirearmDetection/blob/1b59b0416f1a16d0055308842d7fb00c1da83eba/MachineGun_Demo.mp4)

### Hand Gun Detection

![Hand Gun Detection](https://github.com/Anitha-Balachandran/ComputerVision--FirearmDetection/blob/1b59b0416f1a16d0055308842d7fb00c1da83eba/HandGun_Demo.mp4)

## Installation

pip install -r requirements.txt

