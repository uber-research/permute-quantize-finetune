# Permute, Quantize, and Fine-tune

This repository contains the source code and compressed models for the paper
_Permute, Quantize, and Fine-tune: Efficient Compression of Neural Networks_: https://arxiv.org/abs/2010.15703

![Permutation optimization](./imgs/permutation.png?raw=true)

Our method compresses the weight matrices of the network layers by

1) Finding permutations of the weights that result in a functionally-equivalent, yet easier-to-compress network,
2) Compressing the weights using product quantization [[1]](#1-product-quantization),
3) Fine-tuning the codebooks via stochastic gradient descent.

We provide code for compressing and evaluating ResNet-18, ResNet-50 and Mask R-CNN.

## Contents
- [Requirements](#requirements)
- [Data](#data)
- [Training ResNet](#training-resnet)
- [Training Mask R-CNN](#training-mask-r-cnn)
- [Pretrained models](#pretrained-models)
- [Evaluating ResNet](#evaluating-resnet)
- [Evaluating Mask R-CNN](#evaluating-mask-r-cnn)
- [Citation](#citation)
- [References](#references)

## Requirements

Our code requires Python 3.6 or later. You also need these additional packages:

* [PyTorch](https://pytorch.org/)
* [Torchvision](https://github.com/pytorch/vision)
* [Pycocotools](https://pypi.org/project/pycocotools/)
* [Numpy](https://numpy.org/)
* [PyYAML](https://pypi.org/project/PyYAML/)

Additionally, if you have installed [Horovod](https://github.com/horovod/horovod), you may train ResNet with multiple GPUs,
but the code will work with a single GPU even without Horovod.

## Data
Our experiments require either ImageNet (for classification) or COCO (for detection/segmentation).
You should set up a data directory with the datasets.

```bash
<your_data_path>
├── coco
│   ├── annotations   (contains      6 json files)
│   ├── train2017     (contains 118287 images)
│   └── val2017       (contains   5000 images)
└── imagenet
    ├── train         (contains   1000 folders with images)
    └── val           (contains   1000 folders with images)
```

Then, make sure to update the `imagenet_path` or `coco_path` field in the config files to point them to your data.

## Training ResNet

Besides making sure your ImageNet path is set up, make sure to also set up your `output_path` in the config file, or
pass them via the command line:

```bash
python -m src.train_resnet --config ../config/train_resnet50.yaml
```

The `output_path` key inside the config file must specify a directory where all the training output should be saved.
This script will create 2 subdirectories, called `tensorboard` and `trained_models`, inside of the `output_path` directory.

Launching a tensorboard
with the `tensorboard` directory will allow you observe the training state and behavior over time.

```bash
tensorboard --logdir <your_tensorboard_path> --bind_all --port 6006
```

The `trained_models`
directory will be populated with checkpoints of the saved model after initialization, and then after every epoch.
It will also separately store the "best" of these models (the one that attains the highest validation accuracy).

## Training Mask R-CNN

Mask R-CNN (with a ResNet-50 backbone) can be trained by running the command:

```bash
python -m src.train_maskrcnn --config ../config/train_maskrcnn.yaml
```

Once again, you need to specify the `output_path` and the dataset path in the config file before running this.

## Pretrained models

We provide the compressed models we learned from running our code at

```
../compressed_models
```

All models provided have been compressed with k = 256 centroids

|Model (original top-1)              | Regime                       | Comp. ratio | Model size     | Top-1 accuracy (%)|
|:---------------------              |:-----------                  |----------:  |-----------:    | -----------------:|
| ResNet-18 (69.76%)                 | Small blocks<br>Large blocks | 29x<br>43x  |1.54MB<br>1.03MB| 66.74<br>63.33    |
| ResNet-50 (76.15%)                 | Small blocks<br>Large blocks | 19x<br>31x  |5.09MB<br>3.19MB| 75.04<br>72.18    |
| ResNet-50 Semi-Supervised (78.72%) | Small blocks                 | 19x         | 5.09MB         | 77.19             |

We also provide a compressed Mask R-CNN model that attains the following results compared to the uncompressed model:

| Model                 | Size          | Comp. Ratio  | Box AP | Mask AP |
| :------------         | ------------: | -----------: | -----: | ------: |
| Original Mask R-CNN   | 169.4 MB      | -            | 37.9   | 34.6    |
| Compressed Mask R-CNN | 6.65 MB       | 25.5x        | 36.3   | 33.5    |

which you may use as given for evaluation.

## Evaluating ResNet

To evaluate ResNet architectures run the following command from the project root:

```bash
python -m src.evaluate_resnet
```

This will evaluate a ResNet-18 with small blocks by default. To evaluate a ResNet-18 with large blocks, use

```bash
python -m src.evaluate_resnet \
    --model.compression_parameters.large_subvectors True \
    --model.state_dict_compressed ../compressed_models/resnet18_large.pth
```

For ResNet-50 with small blocks, use

```bash
python -m src.evaluate_resnet \
    --model.arch resnet50 \
    --model.compression_parameters.layer_specs.fc.k 1024 \
    --model.state_dict_compressed ../compressed_models/resnet50(_ssl).pth
```

You may load the `resnet50_ssl.pth` model, which has been pretrained on an unsupervised dataset as well.

And for ResNet-50 with large blocks, use

```bash
python -m src.evaluate_resnet \
    --model.arch resnet50 \
    --model.compression_parameters.pw_subvector_size 8 \
    --model.compression_parameters.large_subvectors True \
    --model.compression_parameters.layer_specs.fc.k 1024 \
    --model.state_dict_compressed ../compressed_models/resnet50_large.pth
```

## Evaluating Mask R-CNN

Simply run the command:

```bash
python -m src.evaluate_maskrcnn
```

to load and evaluate the appropriate model.

## Citation

If you use our code, please cite our work:

```
@article{martinez_2020_pqf,
  title={Permute, Quantize, and Fine-tune: Efficient Compression of Neural Networks},
  author={Martinez, Julieta and Shewakramani, Jashan and Liu, Ting Wei and B{\^a}rsan, Ioan Andrei and Zeng, Wenyuan and Urtasun, Raquel},
  journal={arXiv preprint arXiv:2010.15703},
  year={2020}
}
```

## References

###### [1] [Product quantization for nearest neighbor search](https://hal.inria.fr/inria-00514462v2/document)
###### [2] [And the bit goes down: Revisiting the quantization of neural networks](https://arxiv.org/abs/1907.05686)
