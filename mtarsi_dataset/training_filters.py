import matplotlib.pyplot as plt
import numpy as np
import h5py
from keras.models import load_model

main_filepath = "C:/Users/iqras/OneDrive/Documents/NLP_projects/tensorflow_projects/CNN_Course/MTARSI_Dataset"

def convert_to_grid(x_input):
    #shape values of every filter
    number, height, width, channels = x_input.shape  # x_input.shape = (1, 64, 64, 3)

    grid_size = int(np.ceil(np.ceil(np.sqrt(number))))
    grid_height = height * grid_size + (grid_size - 1)
    grid_width = width * grid_size + (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, channels)) + 255

    #traverse through above created grid and plot filters in them
    n = 0
    y_min, y_max = 0, height
    for y in range(grid_size):
        x_min, x_max = 0, width
        for x in range(grid_size):
            if n < number:
                filter_current = x_input[n]
                f_min, f_max = np.min(filter_current), np.max(filter_current)

                grid[y_min:y_max, x_min:x_max] = 255.0 * (filter_current - f_min) / (f_max - f_min)

                n += 1
            x_min += width + 1
            x_max += width + 1
        y_min += height + 1
        y_max += height + 1
    return grid
print()
print("model checkpoint")
model_rgb = load_model(main_filepath + '/model/model_mtarsi_rgb.h5')
model_rgb.load_weights(main_filepath + '/best_weights/Data_Repository/w_1_mtarsi_rgb_255_mean_std.h5')
print("assigned best weights")
assigned_weights = model_rgb.get_weights()
assigned_weights[0] = assigned_weights[0].transpose(3, 0, 1, 2)

grid = convert_to_grid(assigned_weights[0])
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(10, 10)
plt.show()