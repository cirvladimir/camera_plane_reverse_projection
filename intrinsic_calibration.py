import cv2
import numpy as np

from camera import Camera


def intrinsic_calibration(args, camera: Camera):
  print("Use the calibration checkered board and move it around as the images are being captured.")

  # Returns (camera_matrix, distortion)
  checkerboard_size = (args.grid_width - 1,
                       args.grid_height - 1)
  grid_points = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3),
                         np.float32)
  grid_points[0, :, :2] = np.mgrid[0:checkerboard_size[0],
                                   0:checkerboard_size[1]].T.reshape(-1, 2)
  # Creating vector to store vectors of 3D points for each checkerboard image
  world_points = []
  # Creating vector to store vectors of 2D points for each checkerboard image
  image_points = []

  def process_frame(img):
    # Returns true if image was good.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Looking for grid in image...")
    ret, corners = cv2.findChessboardCorners(
        gray, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not ret:
      print("Grid not found, discarding.")
      return False

    # Check if grid points are sharp.

    np_corners = np.array(corners)
    grid_image = img[
        int(min(np_corners[:, :, 1])[0]):int(max(np_corners[:, :, 1])[0]),
        int(min(np_corners[:, :, 0])[0]):int(max(np_corners[:, :, 0])[0]), :]

    image_sharpness = cv2.Laplacian(grid_image, cv2.CV_64F).var()
    if image_sharpness < args.sharpness_threshold:
      print(
          f"Image is blurry, discarding. Sharpness: {image_sharpness} See sharpness_threshold flag."
      )
      return False

    world_points.append(grid_points)
    # refining pixel coordinates for given 2d points.
    refined_corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    image_points.append(refined_corners)

    return True

  num_calibration_images_processed = 0

  while num_calibration_images_processed < args.num_images:
    frame = camera.countdown_capture(delay_seconds=5)

    if process_frame(frame):
      num_calibration_images_processed += 1
      print(
          f"{num_calibration_images_processed}/{args.num_images} images processed"
      )

  print("Image capturing done, calibrating...")
  ret, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
      world_points, image_points, (camera.width, camera.height),
      None, None)

  if not ret:
    print("Some error in cv2.calibrateCamera. Perhaps try again.")
    exit(0)

  np.savez(args.out_file,
           camera_matrix=camera_matrix, distortion=distortion)

  print("Intrinsic calibration done.")

  return (camera_matrix, distortion)
