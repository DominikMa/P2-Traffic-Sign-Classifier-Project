# **Traffic Sign Recognition** 

This repository contains my solution for the project "Traffic Sign Recognition" of the Udacity Self-Driving Car Engineer Nanodegree Program. The python code could be found in the jupyter notebook [P1](P2.ipynb).

The following part of the README contains a writeup which describes how the traffic sign recognition is achieved.


## Writeup

### Goals

In the project the goal is to build, train and analyze a neuronal net which recognizes german traffic signs.
The goal of this project can be seperated in three parts:
* Explore and analyze the data set
* Design, train and test a neuronal net
* Analyze the preditions of the net


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]


### Design and Test of the Model Architecture

### 1. Image augmentation
As a first step, the image data was augmented. This is done to make the calssification of the net more robust. Therefor the following methods are choosen:
1. ### Change Brightness

   The image is converted to the HSV color space. Then the V value is randomly changed.
   ![alt text][image_changed_brightness]

2. ### Crop

   A random 26x26 crop of the image is taken and then resized to 32x32.
   ![alt text][image_cropped]

3. ### Pad

   The image is randomly moved between -8 and 8 pixels independently in x and y direction. The resulting empty space is filled with black.
   ![alt text][image_padded]


4. ### Rotate

   The image is randomly rotated by -15 to 15 degree.
   ![alt text][image_rotated]


This methods should generate images which are likely to appere in the real world.
After the image augmentation the test images set contains 173995 images.


### 2. Preprocessing the image data

In the second step the image data was preprocessed. Therefor the images were converted to the YUV color space and grayscaled by using only the Y value.

After that the image data standardization was preformed by substraction the mean and divede by the standart deviation over all images.

![alt text][grayscaled]


#### 3. The final model architecture

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| 1. Convolution 5x5    | 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| 2. Convolution 3x3    | 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| DROPOUT				|												|
| 3. Convolution 3x3    | 2x2 stride, same padding, outputs 16x16x16 	|
| RELU					|												|
| 4. Convolution 3x3    | 1x1 stride, same padding, outputs 16x16x16 	|
| RELU					|												|
| DROPOUT				|												|
| 5. Convolution 3x3    | 2x2 stride, same padding, outputs 8x8x32	 	|
| RELU					|												|
| 6. Convolution 3x3    | 1x1 stride, same padding, outputs 8x8x32	 	|
| RELU					|												|
| 7. Convolution 3x3    | 1x1 stride, same padding, outputs 8x8x32	 	|
| RELU					|												|
| 8. Convolution 3x3    | 1x1 stride, same padding, outputs 8x8x32	 	|
| RELU					|												|
| DROPOUT				|												|
| Fully connected		| inputs 2048x1, outputs 43x1					|
| Softmax				| 	        									|
|						|												|
|						|												|
 

The model uses a simple architecture which follows the idea of the VGG and ResNet architecture.
It uses manly 3x3 convolutional layer followed by one fully connected layer. For downsampling a stride of 2x2 is used rather the a pooling operation. If the amount of feature maps is increased it is doubled and the picture size halved.

Additionally before the downsampling a dropout layer is used to prevent overfitting.


#### 4. Training of the model

To train the model, the Adam optimizer is used, because it is the preferred optimizer at the moment. As loss the cross entropy between predictions and the correct labels is used.

The Adam paper suggests default settings (0.001) for the learning rate which is used in this training. 

The batch size is choosen as 256. The model was then trained over 50 epochs.

#### 5. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


