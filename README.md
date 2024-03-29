# **Traffic Sign Recognition** 

This repository contains my solution for the project "Traffic Sign Recognition" of the Udacity Self-Driving Car Engineer Nanodegree Program. The python code could be found in the jupyter notebook [Traffic Sign Classifier](Traffic_Sign_Classifier.ipynb). An example run of the notebook with its outputs can be found in [Traffic Sign Classifier Run](Traffic_Sign_Classifier.html)

The following part of the README contains a writeup which describes how the traffic sign recognition is achieved.

## Writeup

### Goals

In the project the goal is to build, train and analyze a neuronal net which recognizes German traffic signs.
The goal of this project can be separated in three parts:
* Explore and analyze the data set
* Design, train and test a neuronal net
* Analyze the predictions of the net


[//]: # (Image References)

[distribution_train]: ./writeup_images/distribution_train.png "Distribution over training images"
[distribution_valid]: ./writeup_images/distribution_valid.png "Distribution over validation images"
[distribution_test]: ./writeup_images/distribution_test.png "Distribution over test images"

[image_changed_brightness]: ./writeup_images/image_changed_brightness.png "Example of an image with changed brightness"
[image_cropped]: ./writeup_images/image_cropped.png "Example of an cropped image"
[image_padded]: ./writeup_images/image_padded.png "Example of an padded image"
[image_rotated]: ./writeup_images/image_rotated.png "Example of an rotated image"

[grayscaled]: ./writeup_images/grayscaled.png "Example of an grayscaled image"

[sign1]: ./images/sign1.png "New Traffic Sign 1"
[sign2]: ./images/sign2.png "New Traffic Sign 2"
[sign3]: ./images/sign3.png "New Traffic Sign 3"
[sign4]: ./images/sign4.png "New Traffic Sign 4"
[sign5]: ./images/sign5.png "New Traffic Sign 5"
[sign6]: ./images/sign6.png "New Traffic Sign 6"
[sign7]: ./images/sign7.png "New Traffic Sign 7"
[sign8]: ./images/sign8.png "New Traffic Sign 8"
[sign9]: ./images/sign9.png "New Traffic Sign 9"
[sign10]: ./images/sign10.png "New Traffic Sign 10"
[sign11]: ./images/sign11.png "New Traffic Sign 11"
[sign12]: ./images/sign12.png "New Traffic Sign 12"
[sign13]: ./images/sign13.png "New Traffic Sign 13"
[sign14]: ./images/sign14.png "New Traffic Sign 14"


### Data Set Summary & Exploration

#### 1. A basic summary of the data set

Using numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset

An interesting aspect of the dataset is the distribution over the labels. It shows how many training, validate and test pictures there are per label in the data sets. This is important because an not uniform distribution could lead to misinterpreting the results. For example a very high amount of stop signs and a low one for yield signs in all datasets could result in a trained model which is good at recognizing stop signs and bad for yield signs but still lead to a very low validate and test error.
Not paying attention to the distribution of the dataset might lead to the conclusion that the model is good for all traffic signs.

The distribution of the datasets is shown in the following pictures:

![alt text][distribution_train]
![alt text][distribution_valid]
![alt text][distribution_test]


### Design and Test of the Model Architecture

### 1. Image augmentation
As a first step, the image data was augmented. This is done to make the classification of the net more robust. Therefor the following methods are chosen:
1. ### Change Brightness

   The image is converted to the HSV color space. Then the V value is randomly changed.
   ![alt text][image_changed_brightness]

2. ### Crop

   A random 24x24 crop of the image is taken and then resized to 32x32.
   ![alt text][image_cropped]

3. ### Pad

   The image is randomly moved between -10 and 10 pixels independently in x and y direction. The resulting empty space is filled with black.
   ![alt text][image_padded]


4. ### Rotate

   The image is randomly rotated by -20 to 20 degree.
   ![alt text][image_rotated]


This methods should generate images which are likely to appear in the real world.
After the image augmentation the test images set contains 173995 images.


### 2. Preprocessing the image data

In the second step the image data was preprocessed. Therefor the images were converted to the YUV color space and grayscaled by using only the Y value.

After that the image data standardization was preformed by subtracting the mean and divide by the standard deviation over all images.

![alt text][grayscaled]


#### 3. The final model architecture

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					|
| Dropout				| keep prop 0.9									| 
| 1. Convolution 5x5    | 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| 2. Convolution 3x3    | 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Dropout				| keep prop 0.5									|
| 3. Convolution 3x3    | 2x2 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| 4. Convolution 3x3    | 1x1 stride, same padding, outputs 16x16x132 	|
| RELU					|												|
| Dropout				| keep prop 0.5									|
| 5. Convolution 3x3    | 2x2 stride, same padding, outputs 8x8x64	 	|
| RELU					|												|
| 6. Convolution 3x3    | 1x1 stride, same padding, outputs 8x8x64	 	|
| RELU					|												|
| 7. Convolution 3x3    | 1x1 stride, same padding, outputs 8x8x64	 	|
| RELU					|												|
| 8. Convolution 3x3    | 1x1 stride, same padding, outputs 8x8x64	 	|
| RELU					|												|
| Dropout				| keep prop 0.5									|
| Fully connected		| inputs 4096x1, outputs 43x1					|
| Softmax				| 	        									|
|						|												|
|						|												|
 

The model uses a simple architecture which follows the idea of the VGG and ResNet architecture.
It uses manly 3x3 convolutional layer followed by one fully connected layer. For downsampling a stride of 2x2 is used rather the a pooling operation. If the amount of feature maps is increased it is doubled and the picture size halved.

Additionally before the downsampling a dropout layer is used to prevent overfitting.


#### 4. Training of the model

To train the model, the Adam optimizer is used, because it is the preferred optimizer at the moment. As loss the cross entropy between predictions and the correct labels is used.

The Adam paper suggests default settings (0.001) for the learning rate which is used in this training. 

The batch size is chosen as 256. The model was then trained over 75 epochs.

#### 5. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93

The final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.972 
* test set accuracy of 0.966

As a starting point the LeNet architecture was tried, because it was already implemented. LeNet preforms well for the MNIST data and therefor may be good at the traffic signs as well.
Simply using the LeNet model resulted in a validation set accuracy about 0.86.

To get a better result first some convolutional layer with RELU activation were added. Together with a longer training time of 20 epochs the model already reached the goal of validation set accuracy to be at least 0.93. 

Recent publications, for example the ResNet paper, showed that the depth of the model is significant for image recognition, so that a model with even more layers was tried.
The ResNet, which is based on the VGG net, is designed for the ImageNet data set and therefor presumably a bit oversized for the German Traffic Sign data set.
It was chosen to follow a mix of the architecture of ResNet and VGG net with just fewer layers.

The architecture follows the rules of using mainly 3x3 convolutional layer, when downsampling use a stride in the convolutional layer and not a pooling operation and when downsampling half picture size and double amount of feature maps. The convolutional layers are then followed by one fully connected layer as classifier.

Because the German Traffic Sign data set is rather small compared to the ImageNet data set additionally dropout layers are used before the downsampling to prevent overfitting.

After these changes the model preformed quite well and was chosen as the final model. It reached a test set accuracy of 0.966.

### Test the Model on New Images

#### 1. 14 new images of traffic signs found on the web

Here are 14 German traffic signs found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3] ![alt text][sign4] ![alt text][sign5]
![alt text][sign6] ![alt text][sign7] ![alt text][sign8] ![alt text][sign9] ![alt text][sign10]
![alt text][sign11] ![alt text][sign12] ![alt text][sign13] ![alt text][sign14]

The ninth, twelfth and fourteenth image might be difficult to classify because they are rotated or tilted.
Especially signs which only appear rarely in the data set, like double curve, end of no passing and slippery road, are chosen because the model should preform well on frequents signs like yield.

#### 2. Predictions on these new traffic signs

Here are the results of the prediction:

| Image			        				|     Prediction       					| 
|:-------------------------------------:|:-------------------------------------:| 
| Right-of-way at the next intersection	| Right-of-way at the next intersection	| 
| Yield     							| Yield 								|
| Yield									| Yield									|
| Keep right	      					| Keep right					 		|
| Stop									| Stop      							|
| Double curve							| Right-of-way at the next intersection	|
| Double curve							| Double curve							|
| End of no passing						| End of no passing						|
| Slippery road							| Road work								|
| Slippery road							| Slippery road							|
| Slippery road							| Slippery road							|
| Bumpy road							| Bumpy road							|
| Bumpy road							| Bumpy road							|
| Dangerous curve to the right			| Slippery road							|

The model was able to correctly guess 11 of the 14 traffic signs, which gives an accuracy of 0.786 This accuracy is significant less then the accuracy of the test set. The reason for this should be the choice of the images from the web. Choosing images which are more frequent in the test set should lead to an higher accuracy.

#### 3. How certain the model is when predicting on each of the new images

For the most images, the model is relatively sure in its prediction, even when the prediction is wrong. The top five soft max probabilities for the first image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  .99        			| Right-of-way at the next intersection			| 
|  .0017   				| Beware of ice/snow 							|
| ~.00					| Children crossing								|
| ~.00	      			| Roundabout mandatory			 				|
| ~.00				    | Double curve     								|

The most top five soft max probabilities look like this.

The only image where the model was unsure was the sixth. Here the model predicted right-of-way at the next intersection on a double curve sign. The double curve appears in the top five soft max probabilities but only with a probability of 0.0022.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  .495        			| Right-of-way at the next intersection			| 
|  .457   				| Beware of ice/snow 							|
|  .026					| Children crossing								|
|  .019	      			| Slippery road					 				|
| ~.00				    | Double curve     								|


