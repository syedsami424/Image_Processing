# import pandas as pd
# import csv
# import os



# labels = []
# for root, dirs, files in os.walk(r"MTARSI 2.v2i.folder\train", topdown=False):
#     for name in dirs:
#         print(name)
#         labels.append(name)
    

# print(labels)
# print(len(labels))

# print()
# print()

# directory_list = []
# for root, dirs, files in os.walk(r"MTARSI 2.v2i.folder\train", topdown=False):
#     for name in dirs:
#         directory_list.append(os.path.join(root, name))

# print(directory_list)

import os
import csv

directory_path = r"MTARSI 2.v2i.folder\train"


image_data = []

for root, dirs, files in os.walk(directory_path):
    for filename in files:
        if filename.lower().endswith(".jpg") and os.path.basename != "train": 

            image_data.append((filename, os.path.basename(root)))
        else:
            print("train encountered")

csv_file = "annotations.csv"
with open(csv_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "class"])  
    writer.writerows(image_data)  

import pandas as pd

df = pd.read_csv(csv_file)
shuffled_df = df.sample(frac=1) 

new_csv_file = "shuffled_image_data.csv"
shuffled_df.to_csv(new_csv_file, index=False)

print(f"Shuffled CSV file '{new_csv_file}' created successfully!")

print(f"CSV file '{csv_file}' created successfully!")
