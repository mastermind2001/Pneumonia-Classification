# Pneumonia Classification using chest X-ray images

##  Overview

A convolutional Neural network (Model.ipynb) is used to build this classifier. A ConvNet takes an image as an input and then differentiate
among different class of images. A ConvNet works on the concept of parameter sharing and thus reduces the number of parameters 
as compared to the standard neural networks.
<br><br>

##  Project Details

The model is implemented using Keras, open source neural-network library written in Python. All the data (images of chest X-ray) 
has converted to hdf('.h5' files) format (using a python package h5py) that you will find in this repo. 

The model has around <b>96% Recall</b>, <b>90% Precision</b> and <b>92% f1 score</b> which is a good measure of accuracy as dataset is imbalanced.

It uses 7 convolution and Max Pool layers along with dropout to prevent overfitting, followed by 3 Dense layers and dropout layers and 'relu' activation function 
and the fourth dense layer contains 1 unit with 'sigmoid' activation function to classify images as having Pneumonia or not.
<br><br>

##  Dataset

The dataset used is avaiable on Kaggle and consists of 5863 images divided into two classes. For the analysis of chest x-ray images, all chest radiographs were 
initially screened for quality control by removing all low quality or unreadable scans.
The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system.

Dataset source : <b>Kaggle</b>


<h5>Copyright &copy; 2020 Akshit Sharma</h5>
