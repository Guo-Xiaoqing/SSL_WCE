# Semi-supervised WCE Image Classification with Adaptive Aggregated Attention 

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/), [Yixuan Yuan](http://www.ee.cityu.edu.hk/~yxyuan/people/people.htm).

## Summary:

### Intoduction:
This repository is for our paper "Semi-supervised WCE Image Classification with Adaptive Aggregated Attention "

### Framework:
To be updated

## Usage:
### Requirement:
Tensorflow 1.4 Or Tensorflow 1.8
Python 3.5

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/SSL_WCE.git
cd SSL_WCE 
```
Use "make_txt.py" to split training data and testing data. The generated txt files are showed in folder "./txt/".
"make_tfrecords.py" is used to make tfrecord format data, which could be stored in folder "./tfrecord/".

WarpDisc2Square.m is the Matlab code for data preprocessing


### Train the model: 
```
sh ./script/train_SSL_WCE.sh
```

### Test the model: 
```
sh ./script/evaluation_SSL_WCE.sh
```

### Well trained model:
You could download the trained SSL_WCE (TensorFlow) from [Google Drive](https://drive.google.com/file/d/1j-Q_u0-Xyp2xYjA55d8zsV1mM9DE2DRc/view?usp=sharing) or 
[Baidu Drive](https://pan.baidu.com/s/1mOAiYFBFTdlW5h7c3f5GWA) (password for download: fas9). Put the model in directory './models'.

## Results:
To be updated

## Citation:
To be updated

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
