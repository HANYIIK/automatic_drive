import numpy as np
import cv2
from skimage.feature import hog
import glob
from sklearn.svm import LinearSVC
import time
import pickle
# 数据前处理时用到
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from scipy.ndimage.measurements import label
# from moviepy.editor import VideoFileClip


# 高斯滤波核大小
blur_ksize = 5
# Canny 边缘检测的 min、max 阈值
canny_min = 50
canny_max = 150

# 霍夫变换参数
rho = 1
theta = np.pi / 180
threshold = 15
min_line_len = 40
max_line_gap = 20


# 图像处理
def process_an_image(image):
    # 1、灰度化/高斯滤波/Canny 边缘检测
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (blur_ksize, blur_ksize), 1)
    edges = cv2.Canny(image_blur, canny_min, canny_max)

    # 2、标记四个坐标点用于 ROI 截取
    # 定义 mask 的四个点
    points = np.array([[(200, 500), (420, 350), (590, 350), (800, 500)]])
    roi_edges = roi_mask(edges, points)

    # 3、霍夫直线提取
    drawing, lines = hough_lines(roi_edges, rho, theta, threshold, min_line_len, max_line_gap)
    draw_lines(image, lines)

    # 4、车道拟合计算
    draw_lanes(drawing, lines)

    result = cv2.addWeighted(image, 0.9, drawing, 0.2, 0)

    return result


def draw_lanes(image, lines):
    # 划分左右车道
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
    if len(left_lines) <= 0 or len(right_lines) <= 0:
        return

    # 清理异常数据
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)

    # 得到左右车道线点的集合，拟合直线
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_results = least_squares_fit(left_points, 325, image.shape[0])
    right_results = least_squares_fit(right_points, 325, image.shape[0])

    # 注意这里点的顺序
    vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])

    # 填充车道区域
    cv2.fillPoly(image, vtxs, (0, 255, 0))


# 迭代计算斜率均值，排除掉与差值差异较大的数据
def clean_lines(lines, threshold_):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(ss - mean) for ss in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold_:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


# 最小二乘法拟合
def least_squares_fit(point_list, ymin, ymax):
    # 最小二乘法拟合
    x = [p[0] for p in point_list]
    y_ = [p[1] for p in point_list]

    # polyfit第三个参数为拟合多项式的阶数，所以1代表线性
    fit = np.polyfit(y_, x, 1)
    fit_fn = np.poly1d(fit)  # 获取拟合的结果

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


def hough_lines(image_gray, rho_, theta_, thresh, min_line_len_, max_line_gap_):
    # 统计概率霍夫直线变换
    lines = cv2.HoughLinesP(image_gray, rho_, theta_, thresh, minLineLength=min_line_len_, maxLineGap=max_line_gap_)

    # 新建白画布一张
    drawing = np.ones((image_gray.shape[0], image_gray.shape[1], 3), dtype=np.uint8)
    return drawing, lines


def draw_lines(image_bgr, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image_bgr, (x1, y1), (x2, y2), [0, 0, 255], 1)


def roi_mask(image, corner_points):
    # 创建掩模
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, corner_points, 255)
    img_masked = cv2.bitwise_and(image, mask)
    return img_masked


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


# 1.图像特征提取
# ① 得到 hog 特征
def get_hog_features(img, orient_, pix_per__cell, cell_per__block, vis=False, feature_vec=True):
    # 可视化
    if vis:
        features, hog_image = hog(img, orientations=orient_,
                                  pixels_per_cell=(pix_per__cell, pix_per__cell),
                                  cells_per_block=(cell_per__block, cell_per__block),
                                  transform_sqrt=True, visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # 不可视化
    else:
        features = hog(img, orientations=orient_,
                       pixels_per_cell=(pix_per__cell, pix_per__cell),
                       cells_per_block=(cell_per__block, cell_per__block),
                       transform_sqrt=True, visualize=vis, feature_vector=feature_vec)
        return features


# ② 原始像素信息特征
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# ③ 计算颜色直方图 hist 特征
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# ④ 整合特征提取
def extract_features(imgs, color__space='RGB', spatial__size=(32, 32),
                     hist__bins=32, orient_=9,
                     pix_per__cell=8, cell_per__block=2, hog__channel=0,
                     spatial__feat=True, hist__feat=True, hog__feat=True):
    # Create a list to append feature vectors to
    feature_image = np.copy(cv2.imread(imgs[0]))
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        img = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color__space != 'RGB':
            if color__space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color__space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color__space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color__space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color__space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        if spatial__feat:
            spatial_features = bin_spatial(feature_image, size=spatial__size)
            file_features.append(spatial_features)
        if hist__feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist__bins)
            file_features.append(hist_features)
        if hog__feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog__channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient_, pix_per__cell, cell_per__block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog__channel], orient_,
                                                pix_per__cell, cell_per__block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# ⑤ 返回单张图片的 1 行特征样本
def single_img_features(img, color__space='RGB', spatial__size=(32, 32),
                        hist__bins=32, hist__range=(0, 256), orient_=9,
                        pix_per__cell=8, cell_per__block=2, hog__channel=0,
                        spatial__feat=True, hist__feat=True, hog__feat=True):
    # 1) Define an empty list to receive features
    feature_image = np.copy(img)
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color__space != 'RGB':
        if color__space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color__space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color__space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color__space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color__space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial__feat:
        spatial_features = bin_spatial(feature_image, size=spatial__size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist__feat:
        hist_features = color_hist(feature_image, nbins=hist__bins, bins_range=hist__range)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog__feat:
        if hog__channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient_, pix_per__cell, cell_per__block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog__channel], orient_,
                                            pix_per__cell, cell_per__block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# 2.窗口分割
# ① 将图像分割为小图
def slide_window(img, x_start_stop=None, y__start_stop=None, xy_window=(256, 256), xy_overlap=(0.6, 0.6)):
    # If x and/or y start/stop positions not defined, set to image size
    if y__start_stop is None:
        y__start_stop = [None, None]
    if x_start_stop is None:
        x_start_stop = [None, None]
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y__start_stop[0] is None:
        y__start_stop[0] = 0
    if y__start_stop[1] is None:
        y__start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y__start_stop[1] - y__start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y__start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# ② 近大远小选择框
def choose_window_size(img, x_start_stop=None, y_start__stop=None, overlap=(0.8, 0.8)):
    if y_start__stop is None:
        y_start__stop = [None, None]
    if x_start_stop is None:
        x_start_stop = [None, None]
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start__stop[0] is None:
        y_start__stop[0] = 0
    if y_start__stop[1] is None:
        y_start__stop[1] = img.shape[0]
    rate = 0
    window_list = []
    start_stop = [[y_start__stop[0], y_start__stop[0] + 64],
                  [y_start__stop[0], y_start__stop[0] + 96],
                  [y_start__stop[0] + 32, y_start__stop[0] + 160],
                  [y_start__stop[0] + 48, y_start__stop[0] + 244]]
    for index in start_stop:
        window_item = slide_window(img, x_start_stop=x_start_stop, y__start_stop=index,
                     xy_window=(64 + rate * 16, 64 + rate * 16), xy_overlap=overlap)
        window_list.append(window_item)
        rate += 1
    return window_list[0] + window_list[1] + window_list[2] + window_list[3]


# ③ 查找小图中的车辆
def search_windows(img, windows_list, clf, scaler, color__space='RGB', spatial__size=(32, 32), hist__bins=32,
                   hist__range=(0, 256), orient_=9,
                   pix_per__cell=8, cell_per__block=2,
                   hog__channel=0, spatial__feat=True,
                   hist__feat=True, hog__feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window_item in windows_list:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window_item[0][1]:window_item[1][1], window_item[0][0]:window_item[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color__space=color__space,
                                       spatial__size=spatial__size, hist__bins=hist__bins, hist__range=hist__range,
                                       orient_=orient_, pix_per__cell=pix_per__cell,
                                       cell_per__block=cell_per__block,
                                       hog__channel=hog__channel, spatial__feat=spatial__feat,
                                       hist__feat=hist__feat, hog__feat=hog__feat)
        # 5) Scale extracted features to be fed to classifier
        test__features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test__features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window_item)
    # 8) Return windows for positive detections
    return on_windows


# ④ 给车画框
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# 颜色转换
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Main Code
# 读取训练图片的位置与文件名
car_filename = glob.glob('vehicles/*/*.png')
num_car_image = len(car_filename)
not_car_filename = glob.glob('non-vehicles/*/*.png')
num_not_car_image = len(not_car_filename)
print('# car images:', num_car_image, '\n# non-car images:', num_not_car_image)

cars = car_filename
notcars = not_car_filename

# 宏定义
color_space = 'YCrCb'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8    # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'     # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
hist_bins = 16   # Number of histogram bins
spatial_feat = True

hist_feat = True    # Histogram features on or off
hog_feat = True     # HOG features on or off
hist_range = (0, 256)
y_start_stop = [300, 700]   # Min and max in y to search in slide_window()

load_model = True

# 训练模型
if load_model:
    with open('model', 'rb') as f:
        svc = pickle.load(f)
    print('SVC model loaded')
    with open('scaler', 'rb') as s:
        X_scaler = pickle.load(s)
    print('X_scaler loaded')
else:
    # 提取特征
    print('正在提取车辆的特征...')
    t1 = time.time()
    car_features = extract_features(cars, color__space=color_space,
                                    spatial__size=spatial_size, hist__bins=hist_bins,
                                    orient_=orient, pix_per__cell=pix_per_cell,
                                    cell_per__block=cell_per_block,
                                    hog__channel=hog_channel, spatial__feat=spatial_feat,
                                    hist__feat=hist_feat, hog__feat=hog_feat)
    t2 = time.time()
    print('提取完成！用时{}秒'.format(round(t2 - t1, 2)))

    print('正在提取非车辆的特征...')
    t1 = time.time()
    notcar_features = extract_features(notcars, color__space=color_space,
                                       spatial__size=spatial_size, hist__bins=hist_bins,
                                       orient_=orient, pix_per__cell=pix_per_cell,
                                       cell_per__block=cell_per_block,
                                       hog__channel=hog_channel, spatial__feat=spatial_feat,
                                       hist__feat=hist_feat, hog__feat=hog_feat)
    t2 = time.time()
    print('提取完成！用时{}秒'.format(round(t2 - t1, 2)))

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # 保存 scaler
    with open('scaler', 'wb') as s:
        pickle.dump(X_scaler, s)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    scaled_X, y = shuffle(scaled_X, y, random_state=rand_state)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    # parameters = {'kernel':('linear', 'rbf'), 'C':[1,10]}
    svc = LinearSVC()
    # clf = grid_search.GridSearchCV(svc, parameters)
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # 保存训练模型
    with open('model', 'wb') as f:
        pickle.dump(svc, f)

# 单张图片测试
# 法一：
# test_features_1 = extract_features(not_car_filename[2000:2001], color__space=color_space,
#                                 spatial__size=spatial_size, hist__bins=hist_bins,
#                                 orient_=orient, pix_per__cell=pix_per_cell,
#                                 cell_per__block=cell_per_block,
#                                 hog__channel=hog_channel, spatial__feat=spatial_feat,
#                                 hist__feat=hist_feat, hog__feat=hog_feat)
# scaled_test_x = X_scaler.transform(test_features_1)
# res = svc.predict(scaled_test_x)
# print(res)

# 法二：
# test_features_2 = [
#     single_img_features(cv2.imread(car_filename[2000]), color__space=color_space, spatial__size=spatial_size,
#                         hist__bins=hist_bins, hist__range=(0, 256), orient_=orient,
#                         pix_per__cell=pix_per_cell, cell_per__block=cell_per_block, hog__channel=hog_channel,
#                         spatial__feat=spatial_feat, hist__feat=hist_feat, hog__feat=hog_feat)]
# scaled_test_x = X_scaler.transform(test_features_2)
# res = svc.predict(scaled_test_x)
# print(res)

test_image = cv2.imread('test6.jpg')
orig_shape = test_image.shape   # 原图的 shape, 用于还原原图的大小

lane_image = cv2.resize(test_image, (960, 540), interpolation=cv2.INTER_CUBIC)
# 画车道线
lane_image = process_an_image(lane_image)
# 还原大小
lane_image = cv2.resize(lane_image, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)

windows = choose_window_size(test_image, x_start_stop=None, y_start__stop=y_start_stop, overlap=(0.6, 0.6))
box = search_windows(test_image, windows, svc, X_scaler, color__space=color_space, spatial__size=spatial_size, hist__bins=hist_bins,
                         hist__range=(0, 256), orient_=orient,
                         pix_per__cell=pix_per_cell, cell_per__block=cell_per_block,
                         hog__channel=hog_channel, spatial__feat=spatial_feat,
                         hist__feat=hist_feat, hog__feat=hog_feat)
drawn_image = draw_boxes(lane_image, box, color=(255, 0, 0), thick=3)
ShowImage('image', drawn_image, 1)
