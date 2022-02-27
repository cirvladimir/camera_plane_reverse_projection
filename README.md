# Camera Plane Reverse Projection

Given a known Z plane inside of a photo, this camera gives you the XY cooridnates on that plane of a pixel on that photo.

It also has scripts for calibrating the camera and defining the plane.

## Requirements

* opencv 4

* python 3

## Usage overview

1. Get camera calibration photos. These find camera focal lengths to correct for lens distortion.

2. Get one photo of aruco marker that defines the plane.

3. Run python3 calibrate.py --image_pattern=/path/to/images/image*.png --output=camera_calibration.npy

4. Run python3 define_plane.py --image=/image/from/step2.png --camera_calibration=camera_calibration.npy --output=camera_transform.npy

5. In your application, use plane_point_finder.py. Initialize it with the two calibration files above. You can even copy-paste plane_point_finder.py in your application.
