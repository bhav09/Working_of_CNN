#building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN
classifier=Sequential()

#step1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#32,3,3 means there will be 32 feature detectors of size 3X3.
#input_shape is for bringing up all the images in the same image size/shape
#3,64,64=3 channels (RGB) 64 is the pixel value range. Can be 256 too but it would be too much for CPU

#step2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3 flattening
classifier.add(Flatten())

#step4 FULL CONNECTION
classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))

#COMPILING
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the cnn to the images
 from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64, 64),batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(training_set,steps_per_epoch=8000,epochs=25,validation_data=test_set,
                    validation_steps=2000)