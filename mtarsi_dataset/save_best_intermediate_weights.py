# This program will train the cnn model again AND save best found weights along with intermediate weights

import matplotlib.pyplot as plt
import os
import numpy as np
import h5py

import visualkeras
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback

main_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset"

model_rgb = []
model_gray = []

for i in range(2):
    model_rgb.append(load_model(main_filepath+"/model/"+"model_mtarsi_rgb.h5"))
    model_gray.append(load_model(main_filepath+"/model/"+"model_mtarsi_gray.h5"))
print("loaded models")
print(model_rgb)
print(model_gray)

epochs = 50
learning_rate = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** (x + epochs), verbose=1)
datasets = ['dataset_mtarsi_rgb_255_mean.hdf5',
            'dataset_mtarsi_rgb_255_mean_std.hdf5',
            'dataset_mtarsi_gray_255_mean.hdf5',
            'dataset_mtarsi_gray_255_mean_std.hdf5']
h = []
for i in range(4):
    with h5py.File(main_filepath+"/preprocessing/Data_Repository/"+datasets[i],'r') as f:
        x_train = f['x_train']
        y_train = f['y_train']
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_validation = f['x_validation']
        y_validation = f['y_validation']
        x_validation = np.array(x_validation)
        y_validation = np.array(y_validation)

    y_train = to_categorical(y_train, num_classes=19)
    y_validation = to_categorical(y_validation, num_classes=19)

    #create best weights file
    best_weights_filepath = 'Data_repository/w_1' + datasets[i][7:-5]+'.h5'

    #schedule to save best weights into above file path
    best_weights = ModelCheckpoint(filepath=best_weights_filepath,
                                    save_weights_only=True,
                                    monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True,
                                    period=1,
                                    verbose=1)

    # Save intermediate weights
    class CustomCallback(Callback):
        def __init__(self):
            self.filename = 0 # file for intermediate weights to be saved
            #self.folder_path = 'Data_repository/'+'intermediate'
            #os.makedirs(self.folder_path, exist_ok=True)

        def on_train_batch_end(self, batch, logs=None):
            if batch % 100 == 0:
                intermediate_weights_filepath = 'Data_repository/intermediate'+'/'+'{0:04d}'.format(self.filename)+"_w_2"+datasets[i][7:-5]+".h5"
                weights_layer_0 = self.model.get_weights()[0]

                with h5py.File(intermediate_weights_filepath, 'w') as f:
                    f.create_dataset("weights_layer_0", data=weights_layer_0, dtype='f')
                    print("\nIntermediate weights saved in:", '{0:04d}'.format(self.filename)+"_w_2"+datasets[i][7:-5]+".h5")

                self.filename += 1

    if i <= 1:
        temp = model_rgb[i].fit(x_train, y_train, batch_size=25, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate, best_weights, CustomCallback()], verbose=1)
        h.append(temp)
        print("Best weights for RGB saved as:", "w_1" + datasets[i][7:-5] + ".h5")
        print()

    elif i >= 2:
        temp = model_gray[i-2].fit(x_train, y_train, batch_size=25, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[learning_rate, best_weights, CustomCallback()], verbose=1)
        h.append(temp)
        print("Best weights for GRAY saved as:", "w_1" + datasets[i][7:-5] + ".h5")
        print()