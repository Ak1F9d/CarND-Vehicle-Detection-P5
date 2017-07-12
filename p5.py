import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import collections

####HOG+SVM###
# Define a function to compute binned color features
def bin_spatial(img, size):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins, bins_range):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def extract_features_single(image, cspace, orient, pix_per_cell,
                     cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, feature_vec=True):
    # apply color conversion if other than 'RGB'
    feature_image = convertColor(image, cspace)
    ## Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    ## Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    ## extract hog features
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(get_hog_features(image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=feature_vec))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    features = np.concatenate((spatial_features, hist_features, hog_features))
    return features
# apply color conversion if other than 'RGB'
def convertColor(image, cspace):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    return feature_image
#  extract features from a list of images
def extract_features(imgs, cspace, orient, pix_per_cell,
                     cell_per_block, hog_channel, spatial_size, hist_bins, hist_range):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # Append the new feature vector to the features list
        features.append(extract_features_single(image, cspace, orient, pix_per_cell,
                     cell_per_block, hog_channel, spatial_size, hist_bins, hist_range))
    # Return list of feature vectors
    return features

cars = glob.glob('vehicles/vehicles/*/*.png')
notcars = glob.glob('non-vehicles/non-vehicles/*/*.png')

#params
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size=(32, 32)
hist_bins=32
hist_range=(0, 1)

# Plot an example of raw and scaled features
car_ind = np.random.randint(0, len(cars))
def visualize_hog(png_path,cspace):
    img=mpimg.imread(png_path)
    feature_image = convertColor(img, cspace)
    plt.figure(dpi=80)
    for channel in range(img.shape[2]):
        hog_feature=get_hog_features(feature_image[:,:,channel],
                            orient, pix_per_cell, cell_per_block,
                            vis=True, feature_vec=True)
        
        plt.subplot(3,4,channel*4+1)
        plt.imshow(feature_image[:,:,channel])
        plt.title('CH%d' % (channel+1), fontsize=8)
        plt.gca().set_axis_off()
        plt.subplot(3,4,channel*4+2)
        plt.imshow(hog_feature[1])
        plt.title('HOG', fontsize=8)
        plt.gca().set_axis_off()
        plt.subplot(3,4,channel*4+3)
        plt.imshow(cv2.resize(feature_image[:,:,channel], spatial_size))
        plt.title('binned features', fontsize=8)
        plt.gca().set_axis_off()
#        plt.subplot(1,4,4)
#        plt.plot(np.histogram(img[:,:,0], bins=hist_bins, range=hist_range)[0])
#        plt.title('CH%d color histogram' % (channel+1))
visualize_hog(cars[car_ind],colorspace)
visualize_hog(notcars[car_ind],colorspace)

t=time.time()
car_features = extract_features(cars, colorspace, orient,
                        pix_per_cell, cell_per_block,
                        hog_channel, spatial_size, hist_bins, hist_range)
notcar_features = extract_features(notcars, colorspace, orient,
                        pix_per_cell, cell_per_block,
                        hog_channel, spatial_size, hist_bins, hist_range)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# Split up data into randomized training and test sets
rand_state = 10 #for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
#visualize normalization
fig = plt.figure(dpi=80)
plt.subplot(121)
plt.plot(X[car_ind])
plt.title('Raw Features')
plt.subplot(122)
plt.plot(scaled_X[car_ind])
plt.title('Normalized Features')
# Use a SVC
svc = LinearSVC()
#parameters = {'C':[0.005, 0.001, 0.0005, 0.0001]}
#clf = GridSearchCV(svc, parameters)
#clf.fit(X_train, y_train)
#clf.best_params_
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))



####sliding window###
def slide_window(img, x_start_stop, y_start_stop,xy_window, xy_overlap):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = img.shape[0]//2
    if y_start_stop[1] == None:
        y_start_stop[1] = int(img.shape[0]*0.9)
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

images_test = glob.glob('test_images/test*.jpg')
def make_windowlist(img):
    window_list=slide_window(img,xy_window=(64, 64), xy_overlap=(0.7, 0.7),x_start_stop=[600, 1000], y_start_stop=[380, 500])
    window_list.extend(slide_window(img,xy_window=(96, 96), xy_overlap=(0.8, 0.8),x_start_stop=[600, 1280],  y_start_stop=[350, 600]))
    window_list.extend(slide_window(img,xy_window=(128, 128), xy_overlap=(0.5, 0.5), x_start_stop=[720, 1280], y_start_stop=[350, 600]))
#    window_list.extend(slide_window(img,xy_window=(192, 192), xy_overlap=(0.7, 0.5),x_start_stop=[1000, 1280],  y_start_stop=[360, 650]))
    return window_list

def detect_car_window(img, feature_vec=True):
    img=img.astype(np.float32)/255
    window_list=make_windowlist(img)
    img_predicts=[]
    features=[]
    for window in window_list:
        img_patch = cv2.resize(img[window[0][1]:window[1][1],window[0][0]:window[1][0]], (64, 64))
        feature=extract_features_single(img_patch, colorspace, orient, pix_per_cell,
                     cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, feature_vec)
        features.append(feature)
    scaled_feature = X_scaler.transform(features)
    img_predicts=svc.predict(scaled_feature)
    heatmap = makeHeatmap(img, window_list, img_predicts)
    return heatmap, window_list, img_predicts

def draw_bboxes(img, predicts, window_list):
    for i in range(len(predicts)):
       if  predicts[i] == 1:
           cv2.rectangle(img,window_list[i][0], window_list[i][1],(0,0,255),6)
    return img
####filtering for false positive###
##heatmap
def makeHeatmap(img, window_list, img_predicts):
    heatmap=np.zeros_like(img, dtype='uint8')
    for i in range(len(window_list)):
        if img_predicts[i]==1:
            heatmap[window_list[i][0][1]:window_list[i][1][1],window_list[i][0][0]:window_list[i][1][0],0]+=1
    return heatmap
##thresholding
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        xLength=bbox[1][0]-bbox[0][0]
        yLength=bbox[1][1]-bbox[0][1]
        if (xLength>25) and (yLength>8) and yLength<xLength*5:
        # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 8)
    # Return the image
    return img

#### test on image ####


for i in range(len(images_test)):
    img =mpimg.imread(images_test[i])
    heatmap, window_list, predicts =detect_car_window(np.copy(img))
    plt.figure(dpi=100)

    plt.subplot(131)
    draw_img=draw_bboxes(np.copy(img), predicts, window_list)
    plt.imshow(draw_img)
    
    plt.subplot(132)
    plt.imshow(heatmap/np.max(heatmap))
    plt.title('max:'+str(np.max(heatmap)))

    heatmap = apply_threshold(heatmap, 3)
    labels = label(heatmap[:,:,0])
    draw_img2 = draw_labeled_bboxes(np.copy(img), labels)
    plt.subplot(133)
    plt.imshow(draw_img2)
    plt.title(str(labels[1]) + 'cars found')
    plt.show()

#### output video ####

def pipeline(img):
    global heatmaps, framecount
    if (framecount%5)==0: #for speed-up and smoothing
        current_heatmap=detect_car_window(np.copy(img),True)[0]
        heatmaps.append(current_heatmap)
    heatmap=sum(heatmaps)
    heatmap_thre = apply_threshold(np.copy(heatmap), 18)
    labels = label(heatmap_thre[:,:,0])
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
#    draw_img = cv2.addWeighted(draw_img, 1, heatmap*6, 0.9, 0)
    framecount+=1
    return draw_img

global heatmaps, framecount
framecount=0
heatmaps = collections.deque(maxlen=6)
#clip1 = VideoFileClip("test_video.mp4").fl_image(pipeline)
clip2 = VideoFileClip("project_video.mp4").fl_image(pipeline)

#check some frames
time_start=15
fps=int(clip2.fps)
for i in range(fps*10):
    time_current=time_start+i/fps
    video_frame=clip2.make_frame(time_current)
    if (framecount%fps)==0:
        plt.imshow(video_frame)
        plt.title(str(time_current))
        plt.show()
#save
#clip1.write_videofile('video_out_test.mp4', audio=False)
#clip2.write_videofile('video_out.mp4', audio=False)
#cv2.imwrite('test9.jpg',cv2.cvtColor(clip2.make_frame(26),cv2.COLOR_RGB2BGR))

#visualize
clip3 = VideoFileClip("project_video.mp4")
plt.figure(figsize=(12,12))
heatmaps2 = collections.deque(maxlen=6)
for i in range(6):
    img =clip3.make_frame(40+i*0.2)
    current_heatmap, window_list, predicts =detect_car_window(np.copy(img))    
    heatmaps2.append(current_heatmap)
    plt.subplot(6,2,i*2+1)
    draw_img=draw_bboxes(np.copy(img), predicts, window_list)
    plt.imshow(draw_img)    
    plt.subplot(6,2,i*2+2)
    plt.imshow(current_heatmap/np.max(current_heatmap))
heatmap=sum(heatmaps2)
heatmap_thre = apply_threshold(np.copy(heatmap), 18)
labels = label(heatmap_thre[:,:,0])
draw_img2 = draw_labeled_bboxes(np.copy(img), labels)
plt.figure()
plt.imshow(labels[0])
plt.figure()
plt.imshow(draw_img2)