import matplotlib.pyplot as plt
import time
import numpy as np
import h5py

import visualkeras
from PIL import ImageFont
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

preprocessing_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset/preprocessing"

with h5py.File(preprocessing_filepath+"/Data_Repository/"+"dataset_mtarsi_rgb_255_mean_std.hdf5", "r") as f:
    print(list(f.keys()))

    x_train = f['x_train']
    y_train = f['y_train']

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_validation = f['x_validation']
    y_validation = f['y_validation']

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    x_test = f['x_test']
    y_test = f['y_test']

    x_test = np.array(x_test)
    y_test = np.array(y_test)

print('Class index from vector:', y_train[0])
y_train = to_categorical(y_train, num_classes = 19)
y_validation = to_categorical(y_validation, num_classes = 19)
print('Class index from matrix:', y_train[0])
#####################################################################################################
model_1 = Sequential()
model_1.add(Conv2D(8, kernel_size = 5, padding = 'same', activation = 'relu', input_shape=(64,64,1)))
model_1.add(MaxPool2D())

model_1.add(Flatten())
model_1.add(Dense(128, activation = 'relu'))
model_1.add(Dense(17, activation='softmax'))

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("MODEL 1 COMPILED")
#####################################################################################################
model_2 = Sequential()
model_2.add(Conv2D(16, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,3)))
#model_2.add(BatchNormalization())
model_2.add(MaxPool2D())

model_2.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
#model_2.add(BatchNormalization())
model_2.add(MaxPool2D())

model_2.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
#model_2.add(BatchNormalization())
model_2.add(MaxPool2D())

model_2.add(Flatten())
model_2.add(Dense(512, activation='relu')) #512 dense neurons is also fine
model_2.add(Dropout(0.5))
model_2.add(Dense(1024, activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(19, activation='softmax'))

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("MODEL 2 COMPILED")
#####################################################################################################
model_3 = Sequential()
model_3.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,1)))
model_3.add(MaxPool2D())

model_3.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model_3.add(MaxPool2D())

model_3.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
model_3.add(MaxPool2D())

model_3.add(Flatten())
model_3.add(Dense(128, activation='relu'))
model_3.add(Dense(17, activation='softmax'))

model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("MODEL 3 COMPILED")
#####################################################################################################
model_4 = Sequential()
model_4.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,1)))
model_4.add(MaxPool2D())

model_4.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model_4.add(MaxPool2D())

model_4.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
model_4.add(MaxPool2D())

model_4.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model_4.add(MaxPool2D())

model_4.add(Flatten())
model_4.add(Dense(128, activation='relu'))
model_4.add(Dense(17, activation='softmax'))

model_4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("MODEL 4 COMPILED")
#####################################################################################################
#plot_model(model_1, to_file='model_1.png', show_shapes=True, show_layer_names=False, rankdir='TB', dpi=500)
font = ImageFont.truetype("arial.ttf", 12)
#visualkeras.layered_view(model=model_1, legend=True, font=font, spacing=20, to_file='model_1.png')
visualkeras.layered_view(model=model_2, legend=True, font=font, spacing=20, to_file='model/model_2.png')
#visualkeras.layered_view(model=model_3, legend=True, font=font, spacing=20, to_file='model_3.png')
#visualkeras.layered_view(model=model_4, legend=True, font=font, spacing=20, to_file='model_4.png')
#####################################################################################################
#print(model_1.summary())

epochs = 25
learning_rate = LearningRateScheduler(lambda x: 1e-3*0.95**(x + epochs), verbose=1)
start_time = time.time()
h_2 = model_2.fit(x_train, y_train, batch_size=20, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate], verbose=1)
print("MODEL 2 TRAINING IS OVER")
stop_time = time.time()

'''
h_1 = model_1.fit(x_train, y_train, batch_size=50, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate], verbose=1)
print("MODEL 1 TRAINING IS OVER")


h_3 = model_3.fit(x_train, y_train, batch_size=50, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate], verbose=1)
print("MODEL 3 TRAINING IS OVER")

h_4 = model_4.fit(x_train, y_train, batch_size=50, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate], verbose=1)
print("MODEL 4 TRAINING IS OVER")
'''
print("Model-2: Training accuracy={0:.5f}, Validation accuracy={1:.5f}".format(max(h_2.history['accuracy']), max(h_2.history['val_accuracy'])))
training_time = stop_time - start_time
print("Time taken:",training_time)

plt.plot(h_2.history['val_accuracy'], '-o')
plt.plot(h_2.history['accuracy'], '-o')
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracies',fontsize=16)

#plt.plot(h_2.history['lr'], '-mo')

plt.show()
'''
print("Model-1: Training accuracy={0:.5f}, Validation accuracy={1:.5f}".format(max(h_1.history['accuracy']), max(h_1.history['val_accuracy'])))
print("Model-3: Training accuracy={0:.5f}, Validation accuracy={1:.5f}".format(max(h_3.history['accuracy']), max(h_3.history['val_accuracy'])))
print("Model-4: Training accuracy={0:.5f}, Validation accuracy={1:.5f}".format(max(h_4.history['accuracy']), max(h_4.history['val_accuracy'])))
'''
