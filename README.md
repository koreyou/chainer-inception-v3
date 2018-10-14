# Using Caffe Pretrained Inception-v3 on Chainer

This project demostrate loading pretrained Inception-v3 on Chainer.

# Prerequisite

## Installation

You need to install dependencies via pip.

```bash
pip install -r requirements.txt
```

## Dataset

Download ImageNet validation dataset from [the official website](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads).

```bash
# Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622
curl http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar | tar -x -C ILSVRC2012_img_val

# Development kit (Task 1 & 2). 2.5MB.
curl http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz | tar -xz
```

## Pretrained model

Download pretrained Caffe model from https://github.com/soeaver/caffe-model/tree/master/cls .
You will see links to the Caffe model hosted on GoogleDrive.

Or I put the direct link [here](https://drive.google.com/open?id=0B9mkjlmP0d7zTEJmNEh6c0RfYzg).

Download and unzip pretrained model on the root directory of the project.


You should end up in the directory like the following.

```
.
├── eval.py
├── ILSVRC2012_devkit_t12
│   ├── ...
│   ├── data
│   │   ├── ILSVRC2012_validation_ground_truth.txt
│   │   └── meta.mat
│   └── evaluation
├── ILSVRC2012_img_val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
│   ├── ...
│   └── ILSVRC2012_val_00050000.JPEG
├── inception-v3
│   ├── deploy_inception-v3-merge.prototxt
│   ├── deploy_inception-v3.prototxt
│   ├── inception-v3.caffemodel
│   └── inception-v3-merge.caffemodel
├── predict.py
├── README.md
└── requirements.txt
```

# Running prediction

Run the following command to output predicted labels in ImageNet format.

```bash
python predict.py \
  inception-v3/inception-v3.caffemodel \
  ILSVRC2012_img_val \
  prediction.txt
```

Evaluate the model performance using ground truth data.

```bash
python eval.py \
  ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt \
  prediction.txt
```

# Result (performance confirmation)

The error rate for ImageNet 2012 validation data was the following.

```
Top-1 error rate: 22.840000%
Top-5 error rate: 6.364000%
```

This corresponds to top-1 (top-5) error of 21.67% (5.75%) report by [Caffe version](https://github.com/soeaver/caffe-model/tree/master/cls) of the same pretrained model, and 23.12% (6.656%) reported by [MXNET version](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md)
The difference is due to the preprocessing.

The prediction took 2 hours and 55 minutes (210 ms per an image) with Intel CPU+iDeep on a laptop.
