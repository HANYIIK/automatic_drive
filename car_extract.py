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


# 显示图像函数
def ShowImage(name_of_image, image_, rate):
    img_min = cv2.resize(image_, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img_min)
    if cv2.waitKey(0) == 27:  # wait for ESC to exit
        print('Not saved!')
        cv2.destroyAllWindows()
    elif cv2.waitKey(0) == ord('s'):  # wait for 's' to save and exit
        cv2.imwrite(name_of_image + '.jpg', image_)  # save
        print('Saved successfully!')
        cv2.destroyAllWindows()


# 加载图片
# 读取图片路径与名称
car_filename = glob.glob('vehicles/*/*.png')
not_car_filename = glob.glob('non-vehicles/*/*.png')
num_car_image = len(car_filename)
not_car_image = len(not_car_filename)
car_image = mping.imread(car_filename[0])
# car_image = cv2.imread(car_filename[0])

print('car images: ', num_car_image)
print('not car images: ', not_car_image)
print('Image shape{} and type {}'.format(car_image.shape, car_image.dtype))
