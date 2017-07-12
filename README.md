
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 15 through 100 of the file called `p5.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car](output_images/image0145.png)
![noncar](output_images/image22.png)   

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:   

car   
![car_features](output_images/car_features.png)  
notcar  
![notcar_features](output_images/notcar_faetures.png)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and here is the best combination:  
 `YCrCb color space`, `orientations=9`, `pixels_per_cell=(8, 8)` , `cells_per_block=(2, 2)`

 Color:  
 In project 4, I learned RGB color space is not robust for brightness variation like shadows. So, I tried other color spaces and YCrCb was the best choice.  

 orientations:  
The greater number of orientations, computing time takes the more. Conversely, the smaller number of orientations, accuracy of prediction deteriorates. I chose 9 as number of orientations, but you can use smaller one if you need more faster processing.

 pixels_per_cell:  
 This is similar to orientations. I chose (8, 8), but you can use greater ones if you need more faster processing.
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
The code for this step is contained in lines 154 through 190 of the file called `p5.py`.  
I trained a linear SVM using HOG features(shape), color histograms(color) and  binned color features(shape & color).
First, I extracted features and normalized them using StandardScaler() function.   
Here is the result of normalizing:
![narmalize](output_images/normalize.png)  

Then, I split features into training and test sets(90% and 10%) and used LinearSVC() function. I gained 0.9899 accuracy score on the test set.  

### Sliding Window Search.....

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The code for this step is contained in lines 195 through 237 of the file called `p5.py`.   
I used three different window sizes: 64x64, 96x96 and 128x128. Each windows overlaps 0.7, 0.8, 0.5.
Here is the windows I used:

![window](output_images/window.png)

I used medium windows (96x96) in right part of the image because the car always go the left lane. Small windows(64x64) were used around center area of the image and large windows(128x128) were used in marginal area because distance from the car is far and the car is small in center.  


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![ex](output_images/ex.png)



---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my video result  
[output_images/video_out.mp4](output_images/video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

##### Here are six frames and their corresponding heatmaps:
![heatmaps](output_images/heatmaps.png)

##### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![label](output_images/label.png)
##### Here the resulting bounding boxes are drawn onto the last frame in the series:
![bboxes](output_images/final_output.png)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here is the problems in my implementation.

1. I used color features so the classifier hard to find cars in the dark environment.

2. It is possible to improve the classifier by additional data augmentation or classifier parameters tuning.

3. The pipeline is not a real-time. It is necessary to optimize the parameters of feature extraction further for increasing the rate.  
4. I assumed the straight and flat load to determine window positions and sizes. So, this pipeline will fail if there are slopes and curves.
