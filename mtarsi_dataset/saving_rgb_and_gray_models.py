import numpy as np
import h5py

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, AvgPool2D

#################################################################################################
rgb_model = Sequential()
rgb_model.add(Conv2D(16, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,3)))
rgb_model.add(MaxPool2D())
#rgb_model.add(Dropout(0.2))

rgb_model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
rgb_model.add(MaxPool2D())
#rgb_model.add(Dropout(0.2))

rgb_model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
#rgb_model.add(BatchNormalization())
rgb_model.add(MaxPool2D())

rgb_model.add(Flatten())
rgb_model.add(Dense(512, activation='relu')) #1024 dense neurons is also fine (more training time with same val accuracy)
rgb_model.add(Dropout(0.5))
rgb_model.add(Dense(1024, activation='relu'))
rgb_model.add(Dropout(0.5))
rgb_model.add(Dense(19, activation='softmax'))

rgb_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("RGB MODEL COMPILED")

rgb_model.save("model/model_mtarsi_rgb.h5")
#################################################################################################
gray_model = Sequential()
gray_model.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,1)))
gray_model.add(MaxPool2D())
#gray_model.add(Dropout(0.2))

gray_model.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
gray_model.add(MaxPool2D())
#gray_model.add(Dropout(0.2))

gray_model.add(Flatten())
gray_model.add(Dense(1024, activation='relu')) #1024 dense neurons is also fine (more training time with same val accuracy)
gray_model.add(Dropout(0.5))
gray_model.add(Dense(19, activation='softmax'))

gray_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("GRAY MODEL COMPILED")

gray_model.save("model/model_mtarsi_gray.h5")