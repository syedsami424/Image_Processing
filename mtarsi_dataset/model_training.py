import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np
import h5py

import visualkeras
from PIL import ImageFont
import tensorflow as tf
from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, AvgPool2D
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

main_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset"

with h5py.File(main_filepath+"/"+"preprocessing"+"/Data_Repository/"+"dataset_mtarsi_rgb_255_mean_std.hdf5", "r") as f:
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
##############################################################################################
model_2 = Sequential()
model_2.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,3)))
model_2.add(MaxPool2D())

model_2.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model_2.add(MaxPool2D())

model_2.add(Flatten())
model_2.add(Dense(512, activation='relu')) #512 dense neurons is also fine
model_2.add(Dropout(0.5))
model_2.add(Dense(1024, activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(19, activation='softmax'))

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("MODEL COMPILED")

#model_2.save('model/model_mtarsi_rgb.h5')

font = ImageFont.truetype("arial.ttf", 12)
visualkeras.layered_view(model=model_2, legend=True, font=font, spacing=20, to_file='model_2.png')

epochs = 20
learning_rate = LearningRateScheduler(lambda x: 1e-3*0.95**(x + epochs), verbose=1)
start = timer()
h_2 = model_2.fit(x_train, y_train, batch_size=20, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate], verbose=1)
print("MODEL TRAINING IS OVER")
end = timer()

print("Model Stats: Training accuracy={0:.5f}, Validation accuracy={1:.5f}".format(max(h_2.history['accuracy']), max(h_2.history['val_accuracy'])))
training_time = "{0:.5f}".format(end - start)
print("Time taken:",training_time)
print()
print(model_2.summary())

plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.facecolor'] = '#303030'
plt.grid(True, linestyle='--', alpha=0.5)

plt.plot(h_2.history['val_loss'], '-o')
plt.plot(h_2.history['loss'], '-o')
legend = plt.legend(['Training loss','Validation loss'], loc='lower right', fontsize='small')
for text in legend.get_texts():
    text.set_color('white')
plt.xlabel('Epoch', fontsize=14, weight="bold")
plt.ylabel('Loss',fontsize=14, weight="bold")
plt.title('Losses',fontsize=14, weight="bold")
plt.savefig('model/Loss.png', dpi=500)
plt.show()

plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.facecolor'] = '#303030'
plt.grid(True, linestyle='--', alpha=0.5)

plt.plot(h_2.history['val_accuracy'], '-o')
plt.plot(h_2.history['accuracy'], '-o')
legend = plt.legend(['Training accuracy','Validation accuracy'], loc='lower right', fontsize='small')
for text in legend.get_texts():
    text.set_color('white')
plt.xlabel('Epoch', fontsize=14, weight="bold")
plt.ylabel('Accuracy',fontsize=14, weight="bold")
plt.title('Accuracies',fontsize=14, weight="bold")
plt.savefig('model/Accuracy.png', dpi=500)
plt.show()
##############################################################################################
#plt.plot(h_2.history['lr'], '-mo')
#plt.show()
'''
# TRAINING MODEL 100 TIMES
a = []
for i in range(1, 100):
    print("ITERATION:",i)
    temp = clone_model(model_2)
    temp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    h_2 = temp.fit(x_train, y_train, batch_size=20, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate], verbose=0)
    print("Model-2: Training accuracy={0:.5f}, Validation accuracy={1:.5f}".format(max(h_2.history['accuracy']), max(h_2.history['val_accuracy'])))
    max_val_acc = max(h_2.history['val_accuracy'])
    a.append(max_val_acc)
    print()

print()
print(a)
print()
'''

'''
BACKUP MODEL:-
model_2 = Sequential()
model_2.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,3)))
model_2.add(AvgPool2D())
model_2.add(Dropout(0.2))

model_2.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model_2.add(AvgPool2D())
model_2.add(Dropout(0.2))

model_2.add(Flatten())
model_2.add(Dense(1024, activation='relu')) #512 dense neurons is also fine
model_2.add(Dropout(0.3))
model_2.add(Dense(40, activation='softmax'))
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("MODEL 2 COMPILED")
'''

'''
ANOTHER BACKUP MODEL(upper one is better)
model_2 = Sequential()
model_2.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(64,64,3)))
model_2.add(Conv2D(8, kernel_size=2, strides=2, padding='same', activation='relu'))
model_2.add(Dropout(0.2))

model_2.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model_2.add(Conv2D(16, kernel_size=2, strides=2, padding='same', activation='relu'))
model_2.add(Dropout(0.2))


model_2.add(Flatten())
model_2.add(Dense(1024, activation='relu'))
model_2.add(Dropout(0.3))
model_2.add(Dense(40, activation='softmax'))

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("MODEL 2 COMPILED")

NOTE: this model NEEDS 1024 layers which in turn increases the time taken to train it.
'''