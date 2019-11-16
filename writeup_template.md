# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/1.jpg "Center image"
[image3]: ./examples/2.jpg "Recovery Image"
[image4]: ./examples/3.jpg "Recovery Image"
[image5]: ./examples/4.jpg "Recovery Image"
[image6]: ./examples/5.jpg "Normal Image"
[image7]: ./examples/6.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 3 and 64 (model.py lines 90-96) 

The model includes RELU layers to introduce nonlinearity (code lines 90, 92, 94, 96, 99, 101 and 103), and the data is normalized in the model using a Keras lambda layer (code line 87). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 97, 100, 102 and 104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 110-114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and I've added new data on curves that the car can't stay on the road with. By using to driving_log_augmented.csv file I added the images again to the data set again if the steering value is not zero.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the architectures used to solve similar problems or to produce a new model from the beginning.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because its structure was simple and did a good job of classifying traffic signs. Later, I tried nvidia architecture and obtained more successful val_loss values.

To increase the data set I flipped the images and add to the dataset (model.py lines 48,49). I used the gradient to get better efficiency from the model (model.py lines 21-27, 43 and 66, drive.py lines 51-93, 108).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I use dropout or pooling layers, use fewer convolution or fewer fully connected layers and collect more data or further augment the data set

Ideally, the model will make good predictions on both the training and validation sets. The implication is that when the network sees an image, it can successfully predict what angle was being driven at that moment.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.Especially on a bend after the bridges, he came out of the road and entered the land. To improve the driving behavior in these cases, I have generated new data about this bend and added it to the dataset. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 85-105) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 65x320x3 Gradient image 						| 
| Convolution 5x5     	| 1x1 stride, no padding, outputs 61x316x24  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 30x158x24 				|
| Convolution 5x5     	| 1x1 stride, no padding, outputs 26x154x36  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x77x36 				|
| Convolution 5x5     	| 1x1 stride, no padding, outputs 9x73x48   	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x36x48  				|
| Convolution 3x3     	| 1x1 stride, no padding, outputs 2x34x64   	|
| RELU					|												|
| Dropout				|0.3											|
| Fully connected		| outputs 100  									|
| RELU					|												|
| Dropout				|0.3											|
| Fully connected		| outputs 50   									|
| RELU					|												|
| Dropout				|0.3											|
| Fully connected		| outputs 10   									|
| Dropout				|0.3											|
| Fully connected		| outputs 1   									|
|						|												|
|						|												|

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go through the middle of the road. These images show what a recovery looks like starting from the right side of the road to the center of the road. :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 66336 number of data points. I then preprocessed this data by lambda layer and cropping top and bottom (model.py lines 87 and 89)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. I used an adam optimizer so that manually training the learning rate wasn't necessary.
