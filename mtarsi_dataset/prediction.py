import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import cv2

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from timeit import default_timer as timer

main_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset"

model_rgb = []
model_gray = []

for i in range(2):
    model_rgb.append(load_model(main_filepath+"/model/model_mtarsi_rgb.h5"))
    model_gray.append(load_model(main_filepath+"/model/model_mtarsi_gray.h5"))

# Assign best weights to model:-
weights = ["w_1_mtarsi_rgb_255_mean.h5",
           "w_1_mtarsi_rgb_255_mean_std.h5",
           "w_1_mtarsi_gray_255_mean.h5",
           "w_1_mtarsi_gray_255_mean_std.h5"]

for i in range(4):
    if i <= 1:
        model_rgb[i].load_weights("best_weights/Data_repository/"+weights[i])
        print("Best RGB weights loaded and assigned", weights[i])

    elif i >= 2:
        model_gray[i-2].load_weights("best_weights/Data_repository/"+weights[i])
        print("Best GRAY weights loaded and assigned", weights[i])

datasets = ["dataset_mtarsi_rgb_255_mean.hdf5",
            "dataset_mtarsi_rgb_255_mean_std.hdf5",
            "dataset_mtarsi_gray_255_mean.hdf5",
            "dataset_mtarsi_gray_255_mean_std.hdf5"]

accuracy_best = 0

for i in range(4):
    with h5py.File(main_filepath+'/'+'preprocessing'+'/Data_Repository/'+datasets[i], 'r') as f:
        x_test = f['x_test']
        y_test = f['y_test']
        x_test = np.array(x_test)
        y_test = np.array(y_test)

    if i == 0:
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)

    if i <= 1:
        temp = model_rgb[i].predict(x_test)
        #if i == 0:
            #print("Predictions shape:", temp.shape)
            #print("Prediction scores:", temp[0])

        temp = np.argmax(temp, axis=1)

        #if i == 0:
            #print("Prediction shape:", temp.shape)
            #print("Predicted indexes:", temp[0:10])
            #print("Actual indexes:", y_test[:10])
            #print()
        accuracy = np.mean(temp == y_test) # Calculate accuracy by comparing prediction with correct class

        if i == 0:
            print("T and F matrix:", (temp == y_test)[0:10])
        print("Testing accuracy: {0:.5f}".format(accuracy))
        print()

    elif i >= 2:
        temp = model_gray[i-2].predict(x_test)
        temp = np.argmax(temp, axis=1)
        accuracy = np.mean(temp == y_test)

        print("Testing Accuracy:{0:.5f}".format(accuracy))
        print()

    if accuracy > accuracy_best:
        accuracy_best = accuracy
        y_predicted_best = temp

    print("Updated best accuracy is:",accuracy_best)

print(classification_report(y_test, y_predicted_best))
c_m = confusion_matrix(y_test, y_predicted_best)
print(c_m)

labels = ['A-10', 'B-1', 'B-2', 'B-29', 'B52', 'Boeing', 'C-5', 'C-17', 'C-21', 'C-130', 'C-135', 'E-3', 'F-15', 'F-16', 'F-22', 'F-35', 'KC-10', 'P-63', 'U-2']

# preprocess test image(s):-
# RGB
with h5py.File(main_filepath+"/preprocessing/Data_Repository/"+"mean_rgb_dataset_mtarsi.hdf5","r") as f:
    mean_rgb = f['mean']
    mean_rgb = np.array(mean_rgb)

with h5py.File(main_filepath+"/preprocessing/Data_Repository/"+"std_rgb_dataset_mtarsi.hdf5","r") as f:
    std_rgb = f['std']
    std_rgb = np.array(std_rgb)

print("Loaded two  RGB arrays:-\n")
print("mean_rgb:", mean_rgb.shape)
print("std_rgb:", std_rgb.shape)
print()
# GRAY
with h5py.File(main_filepath+"/preprocessing/Data_Repository/mean_gray_dataset_mtarsi.hdf5","r") as f:
    mean_gray = f["mean"]
    mean_gray = np.array(mean_gray)

with h5py.File(main_filepath+"/preprocessing/Data_Repository/std_gray_dataset_mtarsi.hdf5","r") as f:
    std_gray = f["std"]
    std_gray = np.array(std_gray)
print("Loaded two GRAY arrays:-\n")
print("mean_gray:", mean_gray.shape)
print("std_gray:", std_gray.shape)

image_custom_bgr = cv2.imread("C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset/test_images/F22.png")

image_custom_rgb = cv2.cvtColor(image_custom_bgr, cv2.COLOR_BGR2RGB)
image_custom_rgb = cv2.resize(image_custom_rgb, (64,64), interpolation=cv2.INTER_CUBIC)

image_custom_rgb_255 = image_custom_rgb / 255.0
image_custom_rgb_255_mean = image_custom_rgb_255 - mean_rgb
image_custom_rgb_255_mean_std = image_custom_rgb_255_mean / std_rgb

# expand dimension of preprocessed image(s) by 1 (number of images)
image_custom_rgb_255_mean = image_custom_rgb_255_mean[np.newaxis, :, :, :]
image_custom_rgb_255_mean_std = image_custom_rgb_255_mean_std[np.newaxis, :, :, :]

image_custom_gray = cv2.cvtColor(image_custom_rgb, cv2.COLOR_RGB2GRAY)
image_custom_gray = image_custom_gray[:, :, np.newaxis]
image_custom_gray_255 = image_custom_gray / 255.0
image_custom_gray_255_mean = image_custom_gray_255 - mean_gray
image_custom_gray_255_mean_std = image_custom_gray_255_mean / std_gray

image_custom_gray_255_mean = image_custom_gray_255_mean[np.newaxis, :, :, :]
image_custom_gray_255_mean_std = image_custom_gray_255_mean_std[np.newaxis, :, :, :]

def bar_chart(scores, bar_title, show_xticks=True, labels=None, counter=0, figsize=(12, 8)):

    plt.figure(figsize=figsize)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['axes.facecolor'] = '#303030'
    plt.grid(True, linestyle='--', alpha=0.5)

    x_positions = np.arange(scores.size)
    barlist = plt.bar(x_positions, scores, align='center', alpha=0.6)
    barlist[np.argmax(scores)].set_color('green')

    if show_xticks:
        plt.xticks(x_positions, labels, rotation=20, fontsize=15)

    plt.xlabel('Class',fontsize=15)
    plt.ylabel('Value',fontsize=15)
    plt.title('Classification: '+ bar_title, fontsize=20)

    plt.savefig(f"prediction_results/result_{counter}.png")
    plt.show()

def plot_image(image, title):
    plt.figure(figsize=(8, 8))
    plt.title(f"Predicted as: {title}", fontsize=16, weight="bold")
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("prediction_results/result.png")
    plt.show()

start = timer()
scores = model_rgb[0].predict(image_custom_rgb_255_mean)
#print(scores)
prediction = np.argmax(scores)
end = timer()

print()
print("RGB (mean) results:-")

#print("Scores             :", scores[0])
#print("Scores sum         :", scores[0].sum())
print("Score of prediction: {0:.5f}".format(scores[0][prediction]))
print("Class index        :", prediction)
print("Label              :", labels[prediction])
print("Time taken         : {0:.5f}".format(end - start))

bar_chart(scores[0], bar_title="rgb_255_mean", show_xticks=True, labels=labels, counter=1)

start = timer()
scores = model_rgb[1].predict(image_custom_rgb_255_mean_std)
end = timer()

prediction = np.argmax(scores)
print()
print("RGB (mean_std) model results:-")
#print("Scores             :", scores[0])
#print("Scores sum         :", scores[0].sum())
print("Score of prediction: {0:.5f}".format(scores[0][prediction]))
print("Class index        :", prediction)
print("Label              :", labels[prediction])
print("Time taken         : {0:.5f}".format(end - start))

bar_chart(scores[0], bar_title="rgb_255_mean_std", show_xticks=True, labels=labels, counter=2)

start = timer()
scores = model_gray[0].predict(image_custom_gray_255_mean)
end = timer()

prediction = np.argmax(scores)
print()
print("GRAY (mean) model results:-")
#print("Scores             :", scores[0])
#print("Scores sum         :", scores[0].sum())
print("Score of prediction: {0:.5f}".format(scores[0][prediction]))
print("Class index        :", prediction)
print("Label              :", labels[prediction])
print("Time taken         : {0:.5f}".format(end - start))

bar_chart(scores[0], bar_title="gray_255_mean", show_xticks=True, labels=labels, counter=3)

start = timer()
scores = model_gray[1].predict(image_custom_gray_255_mean_std)
end = timer()

prediction = np.argmax(scores)
print()
print("GRAY (mean_std) model results:-")
#print("Scores             :", scores[0])
#print("Scores sum         :", scores[0].sum())
print("Score of prediction: {0:.5f}".format(scores[0][prediction]))
print("Class index        :", prediction)
print("Label              :", labels[prediction])
print("Time taken         : {0:.5f}".format(end - start))

bar_chart(scores[0], bar_title="gray_255_mean_std", show_xticks=True, labels=labels, counter=4)
plot_image(image_custom_bgr, labels[prediction])