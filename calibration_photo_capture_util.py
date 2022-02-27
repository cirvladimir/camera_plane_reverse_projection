# This utility heps capture a bunch of camera calibration files.

import argparse
import math
import numpy as np
import cv2
import glob
import time
import threading

parser = argparse.ArgumentParser(
    description='Capture camera calibration photos. All coordinates here are in millimeters.')

parser.add_argument("--video_capture_device_index", default=0, type=int,
                    help="Index of camera. Usually 0 for default opencv setups. Corresponds to cv2.VideoCapture(index).")

parser.add_argument("--calibration_grid_width", default=8, type=int,
                    help="Number of squares along the width of the calibration grid.")

parser.add_argument("--calibration_grid_height", default=10, type=int,
                    help="Number of squares along the height of the calibration grid.")

parser.add_argument("--calibration_sharpness_threshold", default=2000, type=int,
                    help="Discard images which are less sharp than this value. Higher number discards more images, 0 turns feature off.")


parser.add_argument("--x1", default=75.4, type=float,
                    help="X-coordinate of point 1 on aruco calibration image.")

parser.add_argument("--y1", default=254.0, type=float,
                    help="Y-coordinate of point 1 on aruco calibration image.")

parser.add_argument("--x2", default=75.4, type=float,
                    help="X-coordinate of point 2 on aruco calibration image.")

parser.add_argument("--y2", default=477.5, type=float,
                    help="Y-coordinate of point 2 on aruco calibration image.")

parser.add_argument("--num_photos", default=20, type=int,
                    help="Number of camera distortion calibration photos.")

args = parser.parse_args()

vid = cv2.VideoCapture(args.video_capture_device_index)

vid.set(3, 1600)
vid.set(4, 1200)

_, last_frame = vid.read()

new_frame = last_frame
stop_camera = False


def new_frame_updater():
  global new_frame, stop_camera
  while True:
    _, frame = vid.read()
    if frame is not None:
      new_frame = frame

    if stop_camera:
      vid.release()
      break


camera_thread = threading.Thread(target=new_frame_updater)
camera_thread.start()

checkerboard_size = (args.calibration_grid_width - 1,
                     args.calibration_grid_height - 1)
grid_points = np.zeros(
    (1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
grid_points[0, :, :2] = np.mgrid[0:checkerboard_size[0],
                                 0:checkerboard_size[1]].T.reshape(-1, 2)


def process_frame(img):
  # Returns true if image was good.
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  ret, corners = cv2.findChessboardCorners(
      gray, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

  if not ret:
    print("Grid not found, discarding.")
    return False

  # Check if grid points are sharp.

  np_corners = np.array(corners)
  grid_image = img[int(min(np_corners[:, :, 1])[0]):int(max(np_corners[:, :, 1])[0]),
                   int(min(np_corners[:, :, 0])[0]):int(max(np_corners[:, :, 0])[0]),
                   :]

  if cv2.Laplacian(grid_image, cv2.CV_64F).var() < args.calibration_sharpness_threshold:
    print("Image is blurry, discarding. See calibration_sharpness_threshold flag.")
    return False
  return True

  # world_points.append(grid_points)
  # # refining pixel coordinates for given 2d points.
  # refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
  #                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

  # image_points.append(refined_corners)


num_calibration_images_processed = 0

while num_calibration_images_processed < args.num_photos:
  last_frame = new_frame
  print("...")
  cv2.imshow("image", last_frame)

  if process_frame(last_frame):
    num_calibration_images_processed += 1
    print(f"{num_calibration_images_processed}/{args.num_photos} images processed")

  cv2.waitKey(0)
  time.sleep(2)


cv2.destroyAllWindows()


stop_camera = True
camera_thread.join()
