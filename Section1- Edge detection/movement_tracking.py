import tensorflow as tf
import numpy as np
import cv2
from timeit import default_timer as timer
from collections import deque

f1 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) #Sobel filter

cv2.namedWindow('Current view', cv2.WINDOW_NORMAL)
cv2.namedWindow('Contour', cv2.WINDOW_NORMAL)
cv2.namedWindow('Cut fragment', cv2.WINDOW_NORMAL)
cv2.namedWindow('Tracker line', cv2.WINDOW_NORMAL)

#camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("videos/bird.mp4")
h, w = None, None

v = cv2.__version__.split('.')[0]
temp = np.zeros((1080,1920,3), np.uint8)
points = deque(maxlen=50)
counter = 0
fps_start = timer()

while(True):
    _, frame_bgr = camera.read()  #"_" here is a boolean value which specifies wether a frame has been captured or not

    if not _:
        #this condition ends the loop when there are no more frames left to capture.
        #in other words, when the video ends, the loop ends cus of this condition
        break

    if w is None or h is None:
        (h, w) = frame_bgr.shape[:2]

    layer = tf.keras.layers.Conv2D(filters=1,
                            kernel_size=(3,3),
                            strides=1,
                            padding='same',
                            activation='relu',
                            input_shape=(h,w,1),
                            use_bias=False,
                            kernel_initializer=tf.keras.initializers.constant(f1))

    frame_GRAY = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x_input_GRAY = frame_GRAY.reshape(1, h, w, 1).astype(np.float32)

    start = timer()
    output = layer(x_input_GRAY)
    end = timer()
    output = np.array(output[0,:,:,0])
    output = np.clip(output,0,255).astype(np.uint8)

    dilated = output

    if v == '3':
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    else:
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True) #list of contours by sorting from biggest to smallest

    if contours:
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        cv2.rectangle(frame_bgr,(x_min, y_min),(x_min + box_width, y_min + box_height), (0,255,0), 3)
        label = "Person"
        cv2.putText(frame_bgr, label, (x_min - 5, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        center = (int(x_min + box_width / 2), int(y_min + box_height / 2))
        points.appendleft(center)

        cut_fragment_bgr = frame_bgr[y_min + int(box_height * 0.1):y_min + box_height - int(box_height * 0.1), x_min + int(box_width * 0.1): x_min + box_width - int(box_width * 0.1)]

        cv2.imshow('Current view', frame_bgr)
        cv2.imshow('Contour', output)
        cv2.imshow('Cut fragment', cut_fragment_bgr)

        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(temp, points[i - 1], points[i], (0, 255, 0), 3)
            cv2.line(frame_bgr, points[i - 1], points[i], (0, 255, 0), 3)

        cv2.putText(temp, 'X:{0}'.format(center[0]),(50,200), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(temp, label, (100, 200), cv2.FONT_HERSHEY_TRIPLEX, 6, (255,255,255), 6, cv2.LINE_AA)
        cv2.imshow('Tracker line', temp)
        cv2.imshow('Current view', frame_bgr)
        '''
        cv2.putText(temp, "Time :"+"{0:.5f}".format(end - start), (100, 600), cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 4, cv2.LINE_AA)
        cv2.imshow("Spent time",temp)
        '''
    else:
        print("contours = 0")
        cv2.imshow('Current view', frame_bgr)
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0
        cv2.putText(temp, 'No contour found', (100,450), cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 4, cv2.LINE_AA)
        cv2.imshow('Contour', temp)
        cv2.imshow('Cut fragment', temp)
        cv2.imshow('Spent time', temp)

    counter+=1
    fps_stop = timer()

    if fps_stop - fps_start >=1.0:
        print("FPS:",counter)
        counter = 0
        fps_start = timer()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting")
        camera.release()
        cv2.destroyAllWindows()
        break