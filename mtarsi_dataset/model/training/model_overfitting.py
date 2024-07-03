import numpy as np
import matplotlib.pyplot as plt
import h5py

from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler

main_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset"

with h5py.File(main_filepath+"/"+"preprocessing/Data_Repository"+"/"+"dataset_mtarsi_rgb_255_mean_std.hdf5", "r") as f:
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

model = load_model(main_filepath+"/model/"+"model_mtarsi_rgb.h5")
print("loaded model")
print(model.summary())
epochs = 50
learning_rate = LearningRateScheduler(lambda x: 1e-3*0.95**(x + epochs), verbose=1)
h_2 = model.fit(x_train[:100], y_train[:100], batch_size=20, epochs=epochs, validation_data=(x_validation[:500], y_validation[:500]), callbacks=[learning_rate], verbose=1)
print("RGB MODEL OVERFITTING IS OVER")

plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.facecolor'] = '#303030'
plt.grid(True, linestyle='--', alpha=0.5)

plt.plot(h_2.history['accuracy'], '-o')
plt.plot(h_2.history['val_accuracy'], '-o')

legend = plt.legend(['Training accuracy','Validation accuracy'], loc='lower right', fontsize='small')
for text in legend.get_texts():
    text.set_color('white')
plt.xlabel('Epoch', fontsize=14, weight='bold')
plt.ylabel('Accuracy', fontsize=14, weight='bold')
plt.title('Overfitting results(RGB):-', fontsize=16, weight='bold')
plt.savefig('overfitting_results_rgb_military.png', dpi=500)
plt.show()

with h5py.File(main_filepath+"/preprocessing/Data_Repository"+"/dataset_mtarsi_gray_255_mean_std.hdf5", "r") as f:
    print(list(f.keys()))

    x_train = f['x_train']
    y_train = f['y_train']

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_validation = f['x_validation']
    y_validation = f['y_validation']

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    x_test = f["x_test"]
    y_test = f["y_test"]

    x_test = np.array(x_test)
    y_test = np.array(y_test)

print('Class index from vector:', y_train[0])
y_train = to_categorical(y_train, num_classes = 19)
y_validation = to_categorical(y_validation, num_classes = 19)
print('Class index from matrix:', y_train[0])

model = load_model(main_filepath+"/model/"+"model_mtarsi_gray.h5")
print("loaded model")
print(model.summary())
epochs = 50
learning_rate = LearningRateScheduler(lambda x: 1e-3*0.95**(x + epochs), verbose=1)
h_2 = model.fit(x_train[:100], y_train[:100], batch_size=20, epochs=epochs, validation_data=(x_validation[:500], y_validation[:500]), callbacks=[learning_rate], verbose=1)
print("GRAY MODEL OVERFITTING IS OVER")

plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.facecolor'] = '#303030'
plt.grid(True, linestyle='--', alpha=0.5)
plt.plot(h_2.history['accuracy'], '-o')
plt.plot(h_2.history['val_accuracy'], '-o')

legend = plt.legend(['Training accuracy','Validation accuracy'], loc='lower right', fontsize='small')
for text in legend.get_texts():
    text.set_color('white')
plt.xlabel('Epoch', fontsize=14, weight='bold')
plt.ylabel('Accuracy', fontsize=14, weight='bold')
plt.title('Overfitting results(GRAY):-', fontsize=16, weight='bold')
plt.savefig('overfitting_results_gray_military.png', dpi=500)
plt.show()