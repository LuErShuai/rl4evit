This repository is the official PyTorch implementation of [Dynamic Token Pruning for Efficient Vision Transformer via Reinforcement Learning]

## Requirements

Code was tested in virtual environment with Python 3.8. Install requirements as in the requirements.txt file.

## Commands

### Data Preparation

Please prepare the ImageNet dataset into the following structure:

```bash
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...

```

