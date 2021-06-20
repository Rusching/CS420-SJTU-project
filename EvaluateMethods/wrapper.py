import os
from evaluate import evl

image_dir = 'E:/Code_Is_Power/pythonProject/ML/dataset/dataset/32'
label_dir = 'E:/Code_Is_Power/pythonProject/ML/dataset/dataset/test_label'

for i in range(len(os.listdir(image_dir))):
    rand_error, info_error = evl(image_dir, label_dir, i)

