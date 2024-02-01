import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import io

from keras.models import load_model
from timeit import default_timer as timer

preprocessing_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset/preprocessing"
model_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset/model"
best_weights_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset/best_weights"

model = load_model(model_filepath+'/model_mtarsi_gray.h5')
model.load_weights = (best_weights_filepath+"/Data_repository/"+'w_1_mtarsi_gray_255_mean_std.h5')

labels = ['A-10', 'B-1', 'B-2', 'B-29', 'B-52', 'Boeing', 'C-5', 'C-17', 'C-21', 'C-130', 'C-135', 'E-3', 'F-15', 'F-16', 'F-22', 'F-35', 'KC-10', 'P-63', 'U-2']

with h5py.File(preprocessing_filepath+"/Data_Repository/"+"mean_gray_dataset_mtarsi.hdf5", 'r') as f:
    mean_gray = f["mean"]
    mean_gray = np.array(mean_gray)

with h5py.File(preprocessing_filepath+"/Data_Repository/"+"std_gray_dataset_mtarsi.hdf5", 'r') as f:
    std_gray = f["std"]
    std_gray = np.array(std_gray)

def bar_chart(obtained_scores, classes_names):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['axes.facecolor'] = '#303030'
    plt.grid(True, linestyle='--', alpha=0.5)
    x_positions = np.arange(obtained_scores.size)
    bars = plt.bar(x_positions, obtained_scores, align='center', alpha=0.6)
    bars[np.argmax(obtained_scores)].set_color('green')
    plt.xticks(x_positions, classes_names, rotation=10, fontsize=15)
    plt.xlabel('Class', fontsize=15, weight='bold')
    plt.ylabel('Value', fontsize=15, weight='bold')
    plt.title('Obtained Scores:-', fontsize=15, weight='bold')
    plt.tight_layout(pad=2.5)
    b = io.BytesIO()
    plt.savefig(b, format='png', dpi=200)
    plt.close()
    b.seek(0)
    bar_image = np.frombuffer(b.getvalue(), dtype=np.uint8)
    b.close()
    bar_image = cv2.imdecode(bar_image, 1)
    return bar_image

cv2.namedWindow('Current View', cv2.WINDOW_NORMAL)
cv2.namedWindow('Cut fragment', cv2.WINDOW_NORMAL)
cv2.namedWindow('Classified as', cv2.WINDOW_NORMAL)
cv2.namedWindow('Scores', cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(0)
counter = 0
fps_start = timer()
temp = np.zeros((720, 1280, 3), np.uint8)
while True:
    _, frame_bgr = camera.read()
    # detect object:-
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV) # (Hue Saturation Value)
    mask = cv2.inRange(frame_hsv, (0, 130, 80), (180, 255, 255)) # IMPT NOTE: VID-40 on how to find values that fit your project
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # classify detected object:-
    if contours:
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_min + box_width, y_min + box_height), (230, 161, 0), 3)
        cv2.putText(frame_bgr, 'Detected', (x_min - 5, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 161, 0), 2)

        # cut detected fragment:-
        cut_fragment_bgr = frame_bgr[y_min + int(box_height * 0.1):y_min + box_height - int(box_height * 0.1),
                                     x_min + int(box_width * 0.1):x_min + box_width - int(box_width * 0.1)]

        # preprocessing cut fragment:-
        frame_gray = cv2.cvtColor(cut_fragment_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, (64, 64), interpolation=cv2.INTER_CUBIC)
        frame_gray = frame_gray[:, :, np.newaxis]
        frame_gray_255 = frame_gray / 255.0
        frame_gray_255_mean = frame_gray_255 - mean_gray
        frame_gray_255_mean_std = frame_gray_255_mean / std_gray
        frame_gray_255_mean_std = frame_gray_255_mean_std[np.newaxis, :, :,:]

        start = timer()
        scores = model.predict(frame_gray_255_mean_std)
        end = timer()

        prediction = np.argmax(scores)
        cv2.imshow('Current view', frame_bgr)
        cv2.imshow('Cut fragment', cut_fragment_bgr)
        temp[:, :, 0] = 230 # B
        temp[:, :, 1] = 161 # G
        temp[:, :, 2] = 0 # R

        cv2.putText(temp, labels[int(prediction)], (100, 200), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(temp, 'Score: '+'{0:.5f}'.format(scores[0][prediction]), (100, 450), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(temp, 'Time : '+'{0:.5f}'.format(end - start), (100, 600), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)

        cv2.imshow('Classified as: ', temp)
        cv2.imshow('Scores', bar_chart(scores[0], labels))

    else:
        cv2.imshow('Current View', frame_bgr)
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        cv2.putText(temp, 'No object', (100, 450), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.imshow('Cut fragment', temp)
        cv2.imshow('Classified as', temp)
        cv2.imshow('Scores', temp)

    counter += 1
    fps_stop = timer()
    if fps_stop - fps_start >= 1.0:
        print("FPS: ", counter)
        counter = 0
        fps_start = timer()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break