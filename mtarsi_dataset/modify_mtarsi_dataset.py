import pandas as pd
import numpy as np
import h5py
import cv2
import os
from tqdm import tqdm
from sklearn.utils import shuffle

codes_fullpath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset"
image_fullpath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset/train"

labels = ['A-10', 'B-1', 'B-2', 'B-29', 'B52', 'Boeing', 'C-5', 'C-17', 'C-21', 'C-130', 'C-135', 'E-3', 'F-15', 'F-16', 'F-22', 'F-35','KC-10', 'P-63', 'U-2']

print(len(labels))

'''
x_train = np.zeros((1, 64, 64, 3))
y_train = np.zeros(1)

x_temp = np.zeros((1, 64, 64, 3))
y_temp = np.zeros(1)

first_object = True

os.chdir(image_fullpath)

try:
    df = pd.read_csv('annotations.csv')
except FileNotFoundError:
    print("CSV file not found.")
    exit()

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image_name = row['filename']
        class_name = row['class']
        class_index = labels.index(class_name)

        image_array = cv2.imread(image_name)
        if image_array is None:
            raise FileNotFoundError

        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        if image_array.shape[0] >= 64 and image_array.shape[1] >= 64:
            cut_object = cv2.resize(image_array, (64, 64), interpolation=cv2.INTER_CUBIC)

            if first_object:
                x_train[0, :, :, :] = cut_object
                y_train[0] = class_index
                first_object = False
            else:
                x_temp[0, :, :, :] = cut_object
                y_temp[0] = class_index
                x_train = np.concatenate((x_train, x_temp), axis=0)
                y_train = np.concatenate((y_train, y_temp), axis=0)

    except (ValueError, FileNotFoundError):
        continue

x_train, y_train = shuffle(x_train, y_train)

x_temp = x_train[:int(x_train.shape[0] * 0.3), :, :, :]
y_temp = y_train[:int(y_train.shape[0] * 0.3)]

x_validation = x_temp[:int(x_temp.shape[0] * 0.8), :, :, :]
y_validation = y_temp[:int(y_temp.shape[0] * 0.8)]

x_test = x_temp[int(x_temp.shape[0] * 0.8):, :, :, :]
y_test = y_temp[int(y_temp.shape[0] * 0.8):]

os.chdir(codes_fullpath)

with h5py.File('dataset_mtarsi.hdf5', 'w') as f:
    f.create_dataset('x_train', data=x_train, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    f.create_dataset('x_validation', data=x_validation, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    f.create_dataset('x_test', data=x_test, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

print("Dataset creation complete.")
'''