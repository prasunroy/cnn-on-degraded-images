# CNN on Degraded Images
***A study on the effects of different image degradation models on deep convolutional neural network architectures.*** <br />
***The official repository for the work on [Effects of Degradations on Deep Neural Network Architectures](https://arxiv.org/abs/1807.10108).***

<img align='right' height='80' src='https://github.com/prasunroy/hello-world/blob/master/assets/logo.png' />

![badge](https://github.com/prasunroy/cnn-on-degraded-images/blob/master/assets/badge_1.svg)
![badge](https://github.com/prasunroy/cnn-on-degraded-images/blob/master/assets/badge_2.svg)

## Installation
#### Step 1: Install [Python 3.6](https://www.python.org/downloads)
#### Step 2: Install dependencies
```
pip install numpy scipy pandas matplotlib opencv-python tensorflow keras
```
```
pip install git+https://github.com/prasunroy/mlutils.git
```
>For detailed instructions on TensorFlow installation with GPU support refer to the official [*TensorFlow documentation*](https://www.tensorflow.org/install).

## Dataset
### Synthetic Digits
This dataset contains 12,000 synthetically generated images of English digits embedded on random backgrounds. The images are generated with varying fonts, colors, scales and rotations. The backgrounds are randomly selected from a [*subset*](http://images.cocodataset.org/zips/val2017.zip) of [*COCO*](http://cocodataset.org) dataset. The dataset is available at [*Kaggle*](https://www.kaggle.com/prasunroy/synthetic-digits).

Downloading through [*Kaggle API*](https://github.com/Kaggle/kaggle-api) `kaggle datasets download -d prasunroy/synthetic-digits`

![image](https://github.com/prasunroy/cnn-on-degraded-images/blob/master/assets/image_01.png)

### Natural Images
This dataset contains 6,899 images from 8 distinct classes compiled from various sources. The classes include airplane, car, cat, dog, flower, fruit, motorbike and person. The dataset is available at [*Kaggle*](https://www.kaggle.com/prasunroy/natural-images).

Downloading through [*Kaggle API*](https://github.com/Kaggle/kaggle-api) `kaggle datasets download -d prasunroy/natural-images`

![image](https://github.com/prasunroy/cnn-on-degraded-images/blob/master/assets/image_02.png)

## Training Models
>The *`configurations`* section of a train script defines various training parameters. These parameters can be changed by directly modifying the script before training.

#### Training a deep convolutional neural network
```
python train_deepcnn.py
```

#### Training a capsule network
```
python train_capsnet.py
```

## Testing Models
>The *`configurations`* section of a test script defines various testing parameters. These parameters can be changed by directly modifying the script before testing.
```
python test.py
```

## Citation
```
@article{roy2018effects,
  title={Effects of Degradations on Deep Neural Network Architectures},
  author={Roy, Prasun and Ghosh, Subhankar and Bhattacharya, Saumik and Pal, Umapada},
  journal={arXiv preprint arXiv:1807.10108},
  year={2018}
}
```

## Acknowledgements
***This research is supported by [Indian Statistical Institute](https://www.isical.ac.in/) and [NVIDIA GPU Grant Program](https://developer.nvidia.com/academic_gpu_seeding).***
<p align='center'>
  <img height='120' src='https://github.com/prasunroy/cnn-on-degraded-images/blob/master/assets/image_03.png' />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img height='120' src='https://github.com/prasunroy/cnn-on-degraded-images/blob/master/assets/image_04.png' />
</p>

## License
MIT License

Copyright (c) 2018 Prasun Roy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

<br />
<br />

**Made with** :heart: **and GitHub**
