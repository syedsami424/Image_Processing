
MTARSI 2 - v2 2024-05-28 4:18pm
==============================

This dataset was exported via roboflow.com on May 28, 2024 at 2:19 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 20427 images.
Aircrafs are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 240x240 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 40 percent of the image
* Random rotation of between -15 and +15 degrees
* Random Gaussian blur of between 0 and 3.1 pixels
* Salt and pepper noise was applied to 1.93 percent of pixels


