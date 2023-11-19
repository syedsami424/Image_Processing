import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

image1 = cv2.imread('Images/cat.png')
image2 = cv2.imread('Images/horse.jpg')
image3 = cv2.imread('Images/tiger.bmp')

#Grayscale above images:-
image1_g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_g = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image3_g = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

#Apply padding on grayscaled images:-
image1_g_p = np.pad(image1_g,(1, 1), mode='constant', constant_values=0)
image2_g_p = np.pad(image2_g,(1, 1), mode='constant', constant_values=0)
image3_g_p = np.pad(image3_g,(1, 1), mode='constant', constant_values=0)

#create filters to be applied onto the grayscaled padded images:-
#NOTE: we're just comparing which of the these filters is better for edge detection.
filter_1 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])  #Sobel filter
filter_2 = np.array([[0,1,0],[1,-4,1],[0,1,0]])  #Laplacian filter
filter_3 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]) #Prewitt filter

#since above filters havea 3x3 shape, but the images grayscaled padded images have a 2x2 shape, we expand the dimensions
image1_g = np.zeros(image1_g.shape+tuple([3]))
image2_g = np.zeros(image2_g.shape+tuple([3]))
image3_g = np.zeros(image3_g.shape+tuple([3]))

# huge algo for applying the filters:-
for i in tqdm(range(image1_g_p.shape[0]-2)): #853
    for j in range(image1_g_p.shape[1]-2): #1280
        patch = image1_g_p[i:i+3, j:j+3]
        image1_g[i,j,0] = np.sum(patch*filter_1)
        image1_g[i,j,1] = np.sum(patch*filter_2)
        image1_g[i,j,2] = np.sum(patch*filter_3)

for i in tqdm(range(image2_g_p.shape[0]-2)):
    for j in range(image2_g_p.shape[1]-2):
        patch = image2_g_p[i:i+3, j:j+3]
        image2_g[i,j,0] = np.sum(patch*filter_1)
        image2_g[i,j,1] = np.sum(patch*filter_2)
        image2_g[i,j,2] = np.sum(patch*filter_3)

for i in tqdm(range(image3_g_p.shape[0]-2)):
    for j in range(image3_g_p.shape[1]-2):
        patch = image3_g_p[i:i+3, j:j+3]
        image3_g[i,j,0] = np.sum(patch*filter_1)
        image3_g[i,j,1] = np.sum(patch*filter_2)
        image3_g[i,j,2] = np.sum(patch*filter_3)

#remove pixels that are less than 0 and more than 255
'''
image1_g = np.clip(image1_g,0,255)
image2_g = np.clip(image2_g,0,255)
image3_g = np.clip(image3_g,0,255)
'''

figure, ax = plt.subplots(nrows=3, ncols=4)
# note: plotting the images simply as image1,2,3 gave discoloration. using below made it better
ax[0,0].imshow(image1_g, cmap=plt.get_cmap('gray'))
ax[1,0].imshow(image2_g, cmap=plt.get_cmap('gray'))
ax[2,0].imshow(image3_g, cmap=plt.get_cmap('gray'))

ax[0,1].imshow(image1_g[0], cmap = plt.get_cmap('gray'))
ax[1,1].imshow(image2_g[0], cmap = plt.get_cmap('gray'))
ax[2,1].imshow(image3_g[0], cmap = plt.get_cmap('gray'))

ax[0,2].imshow(image1_g[1], cmap = plt.get_cmap('gray'))
ax[1,2].imshow(image2_g[1], cmap = plt.get_cmap('gray'))
ax[2,2].imshow(image3_g[1], cmap = plt.get_cmap('gray'))

ax[0,3].imshow(image1_g[2], cmap = plt.get_cmap('gray'))
ax[1,3].imshow(image2_g[2], cmap = plt.get_cmap('gray'))
ax[2,3].imshow(image3_g[2], cmap = plt.get_cmap('gray'))

for i in range(3):
    for j in range(4):
        ax[i,j].axis('off')
plt.tight_layout()
plt.show()

