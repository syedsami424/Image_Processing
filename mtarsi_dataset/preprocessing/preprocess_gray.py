import numpy as np
import h5py
import cv2

main_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset"

with h5py.File(main_filepath + '/create_dataset/' + 'dataset_mtarsi.hdf5', 'r') as f:

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

x_train = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_train)))
x_validation = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_validation)))
x_test = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_test)))

x_train = x_train[:, :, :, np.newaxis]
x_validation = x_validation[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

print('Numpy arrays of Custom Dataset')
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)
print()

x_train_255 = x_train / 255.0
x_validation_255 = x_validation / 255.0
x_test_255 = x_test / 255.0

with h5py.File(main_filepath+"/"+'preprocessing/Data_Repository' + '/' + 'dataset_mtarsi_gray_255.hdf5', 'w') as f:

    f.create_dataset('x_train', data=x_train_255, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    f.create_dataset('x_validation', data=x_validation_255, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    f.create_dataset('x_test', data=x_test_255, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

mean_gray_dataset_custom = np.mean(x_train_255, axis=0)  # (64, 64, 1)

x_train_255_mean = x_train_255 - mean_gray_dataset_custom
x_validation_255_mean = x_validation_255 - mean_gray_dataset_custom
x_test_255_mean = x_test_255 - mean_gray_dataset_custom

with h5py.File(main_filepath+"/"+'preprocessing/Data_Repository' + '/' + 'mean_gray_dataset_mtarsi.hdf5', 'w') as f:

    f.create_dataset('mean', data=mean_gray_dataset_custom, dtype='f')

with h5py.File(main_filepath+"/"+'preprocessing/Data_Repository' + '/' + 'dataset_mtarsi_gray_255_mean.hdf5', 'w') as f:

    f.create_dataset('x_train', data=x_train_255_mean, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    f.create_dataset('x_validation', data=x_validation_255_mean, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    f.create_dataset('x_test', data=x_test_255_mean, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

std_gray_dataset_custom = np.std(x_train_255_mean, axis=0)  # (64, 64, 1)

x_train_255_mean_std = x_train_255_mean / std_gray_dataset_custom
x_validation_255_mean_std = x_validation_255_mean / std_gray_dataset_custom
x_test_255_mean_std = x_test_255_mean / std_gray_dataset_custom

with h5py.File(main_filepath+"/"+'preprocessing/Data_Repository' + '/' + 'std_gray_dataset_mtarsi.hdf5', 'w') as f:

    f.create_dataset('std', data=std_gray_dataset_custom, dtype='f')

with h5py.File(main_filepath+"/"+'preprocessing/Data_Repository' + '/' + 'dataset_mtarsi_gray_255_mean_std.hdf5', 'w') as f:

    f.create_dataset('x_train', data=x_train_255_mean_std, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    f.create_dataset('x_validation', data=x_validation_255_mean_std, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    f.create_dataset('x_test', data=x_test_255_mean_std, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

print('Original:            ', x_train_255[0, 0, :5, 0])
print('Mean Image:        ', x_train_255_mean[0, 0, :5, 0])
print('Standard Deviation:', x_train_255_mean_std[0, 0, :5, 0])
print()

print('Mean Image:          ', mean_gray_dataset_custom[0, :5, 0])
print('Standard Deviation:  ', std_gray_dataset_custom[0, :5, 0])
print()