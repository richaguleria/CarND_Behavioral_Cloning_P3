
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import img_to_array
import matplotlib 
from PIL import Image
from io import BytesIO

#Load the features and labels from Udacity data and Training data. Augment and Perturb the images with steering > 0.15 and steering < -0.15
samples = []
originalsamples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering = float(line[3])
        originalsamples.append(line)
        if steering > 0.15:
            samples.append(line)
            for i in range(10):
                steering = steering*(1.0 + np.random.uniform(-1,1)/30.0)
                line[3] = steering
                samples.append(line)
        elif steering < -0.15:
            samples.append(line)
            for i in range(10):
                steering = steering*(1.0 + np.random.uniform(-1,1)/30.0)
                line[3] = steering
                samples.append(line)
        else:
            if(steering!=0):
                samples.append(line)
with open('./recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering = float(line[3])
        originalsamples.append(line)
        if steering > 0.15:
            samples.append(line)
            for i in range(10):
                steering = steering*(1.0 + np.random.uniform(-1,1)/30.0)
                line[3] = steering
                samples.append(line)
        elif steering < -0.15:
            samples.append(line)
            for i in range(10):
                steering = steering*(1.0 + np.random.uniform(-1,1)/30.0)
                line[3] = steering
                samples.append(line)
        else:
            if(steering!=0):
                samples.append(line)

#Create Training and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

''' Visualize Dataset '''
#Visualize dataset distribution
def visualiseDataSetDistribution():
    print("Original Samples",len(originalsamples))
    print("Augmented Samples",len(samples))
    m=[]
    samplesForAnalysis = [originalsamples,samples]
    for i in range(2):
        for line in samplesForAnalysis[i]:
            m.append(float(line[3]))
        n = np.array(m)
        nbins = 20
        hist,bins = np.histogram(n,nbins)
        width = 0.8*(bins[1]-bins[0])
        center = (bins[:-1]+bins[1:])/2
        plt.title("samples distribution")
        plt.xlabel("steering angle")
        plt.ylabel("count")
        plt.bar(center,hist,align='center',width=width)
        plt.show()

visualiseDataSetDistribution()

#print some random images, showing images produced during preprocessing and flipped version of preprocessed image for data augmentation
def drawarandomsample():
    for i in range(5):
        j = random.randint(1,2000)
        z = random.randint(0,2)
        name = './data/IMG/'+ samples[j][z].split('/')[-1]
        image = cv2.imread(name)

        fig, axes = plt.subplots(2,3)
        axes[0,0].imshow(image)
        axes[0,0].set_title("Original")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        random_bright = 0.25+np.random.uniform()
        image[:,:,2] = image[:,:,2]*random_bright
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        axes[0,1].imshow(image)
        axes[0,1].set_title("Brightened") 
        
        #convert to YUV for NVIDIA CNN model
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        axes[0,2].imshow(image)
        axes[0,2].set_title("BGR2YUV")
        
        # Crop
        image = image[55:135,:,:]
        axes[1,0].imshow(image)
        axes[1,0].set_title("Cropped")

        # Resize for NVIDIA model
        image = cv2.resize(image,(200,66))
        axes[1,1].imshow(image)
        axes[1,1].set_title("Resized")

        flip = cv2.flip(image,1)
        axes[1,2].imshow(flip)
        axes[1,2].set_title("Flipped")
        
        plt.show()
drawarandomsample()   
 
#Fn. to preprocess images
def ImagePreprocess(image):
    #Increase brightness
    image = img_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    random_bright = 0.25+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    #convert to YUV for NVIDIA CNN model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
    # Crop
    image = image[55:135,:,:]

    # Resize for NVIDIA model
    image = cv2.resize(image,(200,66))

    # Normalisation 
    image = image.astype(np.float32)
    image = image/255.0 - 0.5
    return (image)

#generator function to yield 16 samples at a time for model training 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images , measurements = [] ,[]
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                
                camera = np.random.choice(['center','left','right'])
                
                if camera == 'left':
                    angle += 0.20
                    name = './data/IMG/'+ batch_sample[1].split('/')[-1]
                elif camera == 'right':
                    angle -= 0.20
                    name = './data/IMG/'+ batch_sample[2].split('/')[-1]
                else:
                    name = './data/IMG/'+ batch_sample[0].split('/')[-1]

                image = cv2.imread(name)
                image = ImagePreprocess(image)
                images.append(image)
                measurements.append(angle)
            #Augment the data by flipping
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# NVIDIA Model for model training 
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD, Adam, RMSprop

model = Sequential()
model.add(Convolution2D(24, 5, 5,input_shape=(66,200,3),subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(36, 5, 5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(48, 3, 3, subsample =(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(64, 3, 3,subsample =(1,1)))
model.add(Activation('elu'))
model.add(Convolution2D(64, 3, 3,subsample =(1,1)))
model.add(Flatten())
model.add(Activation('elu'))
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))
model.summary()
model.compile(optimizer=Adam(lr=1e-4), loss='mse')
history_object= model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2,validation_data=validation_generator,
    nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model.h52')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

