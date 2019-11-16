import csv
import cv2
import numpy as np
from matplotlib.pyplot import imread
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

lines = []
with open('data/driving_log_augmented.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
del lines[0]
# split train and validation data
train_data, validation_data = train_test_split(lines, test_size=0.2)

# Include threshold mask functions from the threshold file
from thresholds import *


def thmask(img):
    absolute_sobel_x = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
    color_hls_S = hls_select(img, channel='S', thresh=(90, 255))    
    combined_binary = np.zeros_like(absolute_sobel_x)
    combined_binary[(color_hls_S == 1) | (absolute_sobel_x == 1)] = 1
    combined_img = np.dstack(( combined_binary, combined_binary, combined_binary))*255
    return combined_img
def generator_train(samples, batch_size=64):
    batch_size=int(batch_size/2) # flipped data will be added and the batch will be equal to the first amount
    num_samples = len(samples)
    crc = [0, 0.2, -0.2]    # [center, left, right] camera corrections
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # Add right and left cameras images
            for i in range(3):               
                for batch_sample in batch_samples:                    
                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    image = thmask(cv2.imread(name))
                    angle = float(batch_sample[3])+crc[i]
                    images.append(image)
                    angles.append(angle)
                    # add augmanted data (only flipped data)
                    images.append(cv2.flip(image,1))
                    angles.append(angle*-1.0)          

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)
def generator_test(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []              
            for batch_sample in batch_samples:                    
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                image = thmask(cv2.imread(name))
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
batch_size=64

train_generator = generator_train(train_data, batch_size=batch_size)
validation_generator = generator_test(validation_data, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D

model = Sequential()
# Scale data between -1 and 1
model.add(Lambda(lambda x:x/127.5 - 1., input_shape=(160,320,3)))
# Crop data 
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=24, kernel_size=5, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=36, kernel_size=5, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=48, kernel_size=5, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

#model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(6*len(train_data)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_data)/batch_size), 
            epochs=7, verbose=1)

model.save('model.h5')
