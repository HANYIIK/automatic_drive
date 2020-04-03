import matplotlib.image as mping
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from skimage.feature import hog
import glob
from sklearn.svm import LinearSVC
import pickle
import time
from moviepy.editor import VideoFileClip
# 数据前处理时用到
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# 加载图片
car_filename = glob.glob('./vehicles/*/*/*.png')
not_car_filename = glob.glob('./non-vehicles/*/*/*.png')
num_car_image = len(car_filename)
not_car_image = len(not_car_filename)
car_image = mping.imread(car_filename[0])
print('car images: ', num_car_image)
print('not car images: ', not_car_image)
print('Image shape{} and type {}'.format(car_image.shape, car_image.dtype))
