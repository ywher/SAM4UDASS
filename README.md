# SAM4UDASS: When SAM meets unsupervised domain adaptive segmantic segmentation

This is the official implementation of our paper "SAM4UDASS: When SAM meets unsupervised domain adaptive segmantic segmentation".

## Abstract

Semantic segmentation is an important perception task for intelligent vehicles to understand the surrounding environment. Current deep-learning based methods for semantic segmentation rely on large amounts of labeled data for training, and usually have poor performance in domain shift scenarios. Unsupervised domain adaptation (UDA) methods for semantic segmentation tend to reduce the domain gap and improve the performance of the model in the target domain, where self-training UDA methods currently achieve the state-of-the-art adaptation performance. However, the pseudo-labels generated by self-training methods are still inaccurate, which brings a negative impact on the optimization process of the model. In this paper, we design a simple yet effective method named SAM4UDASS to refine the pseudo-labels by introducing the recently released Segment Anything Model (SAM) into self-training UDA methods. SAM is a foundation model for segmentation task and has strong zero-shot generalization ability, which can generate high quality masks but without specific semantic labels. To circumvent this problem, we use the initial pseudo-labels generated by the teacher network to guide the unlabeled masks from SAM, and design fusion strategies to obtain the refined pseudo-labels for the target domain. SAM4UDASS is a general framework that can be easily integrated into existing self-training UDA methods. Extensive experiments on synthetic-to-real and normal-to-adverse datasets demonstrate the effectiveness of SAM4UDASS. Based on advanced DAFormer, it achieves 71.3%, 65.0%, and 60.5% mIoUs on GTA5-to-Cityscapes, SYNTHIA-to-Cityscapes, and Cityscapes-to-ACDC. Moreover, it also achieves advanced 77.2%, 69.3%, and 69.5% mIoUs on the three settings, respectively, when using MIC.

## Method Overview

## Environment Setup

## Dataset Preparation

## Usage Example

## Acknowledgement

## Citation

> TBD

## Concate

Weihao Yan: weihao_yan@outlook.com
