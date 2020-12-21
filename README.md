# Semi-supervised WCE Image Classification with Adaptive Aggregated Attention 

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/), [Yixuan Yuan](http://www.ee.cityu.edu.hk/~yxyuan/people/people.htm).

## Summary:

### Intoduction:
This repository is for our MIA 2020 paper ["Semi-supervised WCE Image Classification with Adaptive Aggregated Attention"](https://www.sciencedirect.com/science/article/pii/S1361841520300979)

### Framework:
![](https://github.com/Guo-Xiaoqing/SSL_WCE/raw/master/Figs/network.png)

## Usage:
### Requirement:
Tensorflow 1.4 (1.8)
Python 3.5

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/SSL_WCE.git
cd SSL_WCE 
```
* Use "make_txt.py" to split training data and testing data. The generated txt files are showed in folder "./txt/".

* "make_tfrecords.py" is used to make tfrecord format data, which could be stored in folder "./tfrecord/".

* Note that we implemented data augmentation before training and then used the augmented dataset for training in this paper. However, it may be inconvenient for you. Therefore, to avoid additional data augmentation before training, online data augmentations, including random flips and rotations, are added in script "utilsForTF.py". 

* WarpDisc2Square.m is the Matlab code for data preprocessing. 
<img src="https://github.com/Guo-Xiaoqing/SSL_WCE/raw/master/Figs/preprocess.png" width="600" height="175" />

### Train the model: 
```
sh ./script/train_SSL_WCE.sh
```

* In this paper, the backbone of our method is DenseNet. Besides, we also integrate our method on other classification backbones: VGG, ResNet-50 and ResNeXt-50. These models are in the folder "./nets/". Several changes, such as decreasing the number of layers and feature maps, are made to ensure the size of these backbone networks to be similar to that of DenseNet. The hyperparameters utilized to these backbones are written in their corresponding model scripts.

### Test the model: 
```
sh ./script/evaluation_SSL_WCE.sh
```

### Well trained model:
You could download the trained SSL_WCE model from [Google Drive](https://drive.google.com/file/d/1j-Q_u0-Xyp2xYjA55d8zsV1mM9DE2DRc/view?usp=sharing). Put the model in directory './models'.

### Results:
* Attention maps and inputs of the second branch derived from validation samples are shown in [Results](https://github.com/Guo-Xiaoqing/SSL_WCE/tree/master/models/attention_map/).

* Log files recorded with tensorflow 1.4 and 1.8 are listed in [Logs](https://github.com/Guo-Xiaoqing/SSL_WCE/tree/master/models/logs/), which report the running time, loss and accuracy of a mini-batch during training phase.

## Citation:
@article{guo2020semi,
  title={Semi-supervised WCE Image Classification with Adaptive Aggregated Attention},
  author={Guo, Xiaoqing and Yuan, Yixuan},
  journal={Medical Image Analysis},
  pages={101733},
  year={2020},
  publisher={Elsevier}
}

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
