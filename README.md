# A PyTorch Implementation of DSCCP

Dynamic Strip Convolution Characterized Plugin for Medical Anatomy Segmentation**Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation**.
Guyue Hu, Yukun Kang, Gangming Zhao, Zhe Jin, Chenglong Li, and Jin Tang

### **Abstract**

Medical anatomy segmentation is essential for computer-aided diagnosis and lesion localization in medical images. For example, segmenting individual ribs benefits localizing the lung lesions and providing vital medical measurements (such as rib spacing) for generating medical reports. Existing methods segment shape-different anatomies (such as striped ribs, bulky lungs, and angular scapula) with the same network architecture, the morphology heterogeneity is heavily overlooked. Although some shape-aware operators like deformable convolution and dynamic snake convolution have been introduced to cater to specific object morphology, they still struggle with orientation-varying strip structures, such as 24 ribs and 2 clavicles. In this paper, we propose a novel Dynamic Strip Convolution Characterized Plugin (DSCCP) for medical anatomy segmentation, which is comprised of dynamic strip convolution (DSC) operator and adaptive morphology perception (AMP) strategy. Specifically, the dynamic strip convolution customizes gradually varying directions and offsets for each local region, achieving dynamic striped receptive fields. Additionally, the adaptive morphology perception strategy incorporates insights from various shape-aware convolutional kernels, enabling the model to discern and integrate crucial representations corresponding to heterogeneous anatomies. Extensive experiments on two large-scale datasets demonstrate the effectiveness and superiority of the proposed approach for tackling heterogeneous medical anatomy segmentation.

## Features

#### 1. Dataset

- [X]  Synapse
- [ ]  DRIVE

#### 2. Tasks

- [X]  2D Medical Anatomy Segmentation
- [ ]  3D Medical Anatomy Segmentation

#### 3. Visualization

* [ ]  CXRS
* [ ]  Synapse

## Prerequisites

Our code is based on **Python3.5**. There are a few dependencies to run the code in the following:

- Python == 3.10
- **PyTorch >= 2.0.0**
- Tensorboard
- Other version info about some Python packages can be found in `requirements.txt`

## Usage

#### Data preparation

##### Synapse

Download the Synapse dataset from ([https://drive.google.com/file/d/115-vkjCapans\_Mx3EXLxZsxr\_WSbpXxm/view?usp=sharing](https://drive.google.com/file/d/115-vkjCapans_Mx3EXLxZsxr_WSbpXxm/view?usp=sharing))

##### Other Datasets

Not supported now.

#### Training

```To
python train.py --dataset Syanpse --root_path your DATA_DIR --max_epochs 400 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

#### Testing

To run the testing procedure, you should run the command line below

```commandline
python test.py --dataset Synapse --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 400 --base_lr 0.05 --img_size 224 --batch_size 24
```

## Results

#### Table

TODO

#### Visualization

TODO

## Reference

[1] Qi Y, He Y, Qi X, et al. Dynamic snake convolution based on topological geometric constraints for tubular structure segmentation. ICCV 2023.

[2] [BRAU-Net++](https://github.com/Caipengzhou/BRAU-Netplusplus): referred for some code of Synpase dataset processing.
