# Camera and Lidar 3D Object Tracking

This project encorporates
1. Keypoint detectors, descriptors, and methods to match them between successive images
2. Detecting objects in an image using deep learning via YOLOv3. 
3. Mapping camera image data with Lidar points in 3D space to identify 3D objects.
4. Match 3D objects over time by using keypoint correspondences.
5. Compute a time-to-collision (TTC) with objects based on Lidar measurements.
6. Compute a time-to-collision (TTC) with objects based on Camera measurements.
7. Tests different combinations of detectors and descrtiptors to find most suitable configuration.

The overall project schematic is shown below based on Udacity SFND.

<img src="images/course_code_structure.png" width="779" height="414" />

## Dependencies for Running Locally

* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Download missing yolov3.weights file from https://pjreddie.com/media/files/yolov3.weights and copy to dat/yolo/ folder.
3. Make a build directory in the top level project directory: `mkdir build && cd build`
4. Compile: `cmake .. && make`
5. Run it: `./3D_object_tracking`.# Camera_Lidar_3D_Object_Tracking
