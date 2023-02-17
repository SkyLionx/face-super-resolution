# 🧑 Face Recognition with Super Resolution

This project implements a **deep learning model** performing **face recognition** by using **super-resolution** techniques in order to enhance images of faces acquired by a camera with a very low resolution or from a long distance. 
Our hypothesis is that increasing the images resolution, we can leverage more information and build a model which can perform better in the face recognition task w.r.t. a model which uses the low resolution images.

We describe and evaluate **several methods** to perform the upscaling and compare them with a base model without using super-resolution. Moreover, we propose and test two models, which are **Generative Adversarial Networks (GANs)**, able to perform upscaling from images with a resolution lower than the one used by the most popular state-of-the-art models.

The proposed system performs the **open set identification task** and its architecture is as follows:

<div align="center">

![architecture](https://user-images.githubusercontent.com/23276420/219739645-11dd3ca2-1e78-4cf2-a226-bc69055cd1cd.png)

</div>

To perform the face **localization task**, two different techniques are compared:
- Haar Cascade classifier implemented following the Viola-Jones algorithm
- Multi-Task Cascaded Convolutional Neural Network (MTCNN) model

The cropped faces are then upscaled from 32×32 to 128×128 using and comparing 6 different approaches:
- OpenCV Resize using bilinear interpolation
- Enhanced Deep Super-Resolution (EDSR)
- Super-Resolution Generative Adversarial Network
- Enhanced SRGAN (ESRGAN)
- **Our baseline GAN model**
- **Our improved GAN model using Edge Detection**

Finally, the upscaled faces are processed by **our simple Face Recognition model** based on the **ResNet architecture**, which has been implemented just to compare the results of a baseline model by using the different versions of the input images.

In order to train and test the models two different datasets were used:
- [CASIA-WebFace](https://arxiv.org/abs/1411.7923v1) for training
- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw) for testing

An **interactive Colab Notebook** is available in order to follow the whole dataset processing, model training and evaluation.

<a href="https://colab.research.google.com/github/SkyLionx/face-super-resolution/blob/main/Face%20Recognition%20with%20Super%20Resolution.ipynb" target="_blank">
<img src="https://img.shields.io/badge/Colab-Open%20Notebook-green?style=for-the-badge&logo=googlecolab&color=blue">
</a>
<br /><br />

Moreover, a full [`Report`](https://github.com/SkyLionx/face-super-resolution/blob/main/Report.pdf) and a PDF [`Presentation`](https://github.com/SkyLionx/face-super-resolution/blob/main/Presentation.pdf) are available in the repo.

## Results

### Face detection
For the face detection task, we took in consideration both the qualitative results obtained and the processing speed of the two methods.
The results obtained by the Haar Cascade Classifier and MTCNN  are comparable, while we measured that the time required in order to process and extract faces from our dataset is much less using the first one.
For this reason, at the end we decided to opt for the faster method since we don’t lose too much in accuracy and we can save precious processing time.

### Super Resolution

In the following image we present a comparison of the results we obtained using the different super resolution techniques:
<div align="center">

![super-res-comparison](https://user-images.githubusercontent.com/23276420/219738969-8a8c2e6f-6045-42c9-9d8a-4eb15e728c10.png)

</div>

### Face Recognition
As final results, here we present the metrics achieved by our simple Face Recognition module by comparing the performance using different images as input.
Original images are raw images contained in the dataset both in SR (Original-128) and LR (Original-32). Then there is a comparison between our two trained GANs, followed by a simple Bilinear Interpolation upscaling and finally comparing them to the VGG-Face state-of-the-art model.

|                            | Recognition Rate | DIR@5 | DIR@15 | Genuine Recognition Rate (GRR) | Equal Error Rate (ERR) | Best Threshold |
|----------------------------|:----------------:|:-----:|:------:|:------------------------------:|:----------------------:|:--------------:|
| **Original-128**           |       0,37       |  0,44 |  0,47  |              0,13              |          0,63          |      0,25      |
| **Original-32**            |       0,03       |  0,07 |  0,12  |              0,04              |          0,97          |      0,20      |
| **Canny-GAN**              |       0,32       |  0,40 |  0,42  |              0,15              |          0,68          |      0,25      |
| **GAN**                    |       0,32       |  0,40 |  0,42  |              0,16              |          0,68          |      0,25      |
| **Bilinear Interpolation** |       0,31       |  0,38 |  0,40  |              0,20              |          0,68          |      0,25      |
| **VGG-Face**               |       0,40       |  0,44 |  0,46  |              0,26              |          0,60          |      0,20      |

## Contributors

<a href="https://github.com/SkyLionx" target="_blank">
  <img src="https://img.shields.io/badge/Profile-Fabrizio%20Rossi-green?style=for-the-badge&logo=github&labelColor=blue&color=white">
</a>
<br/><br/>
<a href="https://github.com/dotmat3" target="_blank">
  <img src="https://img.shields.io/badge/Profile-Matteo%20Orsini-green?style=for-the-badge&logo=github&labelColor=blue&color=white">
</a>

## Technologies

In this project the following Python libraries were adopted:
- TensorFlow
- OpenCV for the Haar Cascade Classifier
- Numpy 
- Matplotlib for plotting
