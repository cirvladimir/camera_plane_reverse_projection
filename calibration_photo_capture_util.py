# This utility heps capture a bunch of camera calibration files.

import argparse
import math
from cv2 import undistort
import numpy as np
import cv2
from cv2 import aruco
import glob
import time
import threading
import os
import reverse_projection

parser = argparse.ArgumentParser(
    description='Capture camera calibration photos. All coordinates here are in millimeters.'
)

parser.add_argument(
    "--skip_intrinsic_calibration",
    action='store_true',
    help="Do not calibrate focal lengths/distortions. Load ones from a file instead. You should do this if you've calibrated the same camera before and are just repositioning it, without changing the focus or zoom."
)

parser.add_argument("--skip_extrinsic_calibration",
                    action='store_true',
                    help="Skip calibrating camera position and only calibrate intrinsic parameters.")

parser.add_argument("--test_only",
                    action='store_true',
                    help="Test an existing calibration file.")

parser.add_argument(
    "--intrinsic_params_file",
    help="Where to write/read intrinsic camera parameters. Recommend .npz extension. Write/read based on calibrate_intrinsic flag. Optional if you specify all_params_file."
)

parser.add_argument(
    "--all_params_file",
    help="Where to write extrinsic + intrinsic camera parameters. Recommend .npz extension. Only created with calibrate_extrinsic."
)

parser.add_argument(
    "--camera_index",
    default=0,
    type=int,
    help="Index of camera. Usually 0 for default opencv setups. Corresponds to cv2.VideoCapture(index)."
)

parser.add_argument(
    "--calibration_grid_width",
    default=8,
    type=int,
    help="Number of squares along the width of the calibration grid.")

parser.add_argument(
    "--calibration_grid_height",
    default=10,
    type=int,
    help="Number of squares along the height of the calibration grid.")

parser.add_argument(
    "--calibration_sharpness_threshold",
    default=50,
    type=int,
    help="Discard images which are less sharp than this value. Higher number discards more images, 0 turns feature off."
)

parser.add_argument("--camera_position_photo",
                    help="Optional. Photo of aruco markers for positioning camera. If unspecified, script will prompt you to take a photo with a camera.")

parser.add_argument("--x1",
                    default=185.0,
                    type=float,
                    help="X-coordinate of point 1 on aruco calibration image. Only needed if you're using camera_position_photo.")

parser.add_argument("--y1",
                    default=12.0,
                    type=float,
                    help="Y-coordinate of point 1 on aruco calibration image. Only needed if you're using camera_position_photo.")

parser.add_argument("--x2",
                    default=182.5,
                    type=float,
                    help="X-coordinate of point 2 on aruco calibration image. Only needed if you're using camera_position_photo.")

parser.add_argument("--y2",
                    default=235.5,
                    type=float,
                    help="Y-coordinate of point 2 on aruco calibration image. Only needed if you're using camera_position_photo.")

parser.add_argument("--num_photos",
                    default=20,
                    type=int,
                    help="Number of camera distortion calibration photos.")

parser.add_argument("--start_delay",
                    default=0,
                    type=float,
                    help="Number of seconds to wait at the start.")

parser.add_argument("--debug",
                    action='store_true',
                    help="Print extra debug info.")

args = parser.parse_args()

# Minimal check for argument validity.
if not args.test_only:
  if not args.skip_extrinsic_calibration:
    if args.all_params_file is None:
      print(
          "You must specify --all_params_file=<path to new file> for storing"
          " parameters."
      )
      exit(0)

    if args.skip_intrinsic_calibration:
      # Check that some params file exists
      if args.intrinsic_params_file is not None:
        if not os.path.exists(args.intrinsic_params_file):
          print(
              "If you're skipping intrinsic calibration, intrinsic_params_file should exist.")
          exit(0)
      else:
        if not os.path.exists(args.all_params_file):
          print("If you're skipping intrinsic calibration, all_params_file should exist, or specify intrinsic_params_file.")
          exit(0)

  if args.skip_extrinsic_calibration and not args.skip_intrinsic_calibration:
    if args.intrinsic_params_file is None:
      print("You need to specify intrinsic_params_file if you're skipping extrinsic calibration.")
      exit(0)
else:
  if args.all_params_file is None:
    print("You're using the test_only flag. You must specify the file to test in all_params_file.")
    exit(0)
  if not os.path.exists(args.all_params_file):
    print("Specified all_params_file does not exist.")
    exit(0)

if args.camera_position_photo is None:
  vid = cv2.VideoCapture(args.camera_index)

  vid.set(3, 3264)

  _, last_frame = vid.read()

  new_frame = last_frame
  stop_camera = False
  show_camera = False
  preview_display_image = None

  def new_frame_updater():
    global new_frame, stop_camera
    preview_window_created = False
    while True:
      _, frame = vid.read()
      if frame is not None:
        new_frame = frame

        if show_camera:
          if not preview_window_created:
            cv2.namedWindow("live_preview", cv2.WINDOW_NORMAL)
            preview_window_created = True

          if preview_display_image is not None:
            cv2.imshow("live_preview", preview_display_image)
          else:
            cv2.imshow("live_preview", new_frame)
          cv2.waitKey(1)

      if stop_camera:
        if preview_window_created:
          cv2.destroyWindow("live_preview")
        vid.release()
        break

  show_camera = True

  camera_thread = threading.Thread(target=new_frame_updater)
  camera_thread.start()


def countdown(n):
  for i in range(n):
    print(n - i)
    time.sleep(1)


def calibrate_intrinsic():
  print("Use the calibration checkered board and move it around as the images are being captured.")

  # Returns (camera_matrix, distortion)
  checkerboard_size = (args.calibration_grid_width - 1,
                       args.calibration_grid_height - 1)
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
    if image_sharpness < args.calibration_sharpness_threshold:
      print(
          f"Image is blurry, discarding. Sharpness: {image_sharpness} See calibration_sharpness_threshold flag."
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

  while num_calibration_images_processed < args.num_photos:
    countdown(5)
    last_frame = new_frame

    if process_frame(last_frame):
      num_calibration_images_processed += 1
      print(
          f"{num_calibration_images_processed}/{args.num_photos} images processed"
      )

  print("Image capturing done, calibrating...")
  ret, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
      world_points, image_points, (last_frame.shape[1], last_frame.shape[0]),
      None, None)

  if not ret:
    print("Some error in cv2.calibrateCamera. Perhaps try again.")
    exit(0)

  if args.intrinsic_params_file is not None:
    np.savez(args.intrinsic_params_file,
             camera_matrix=camera_matrix, distortion=distortion)
  elif args.all_params_file is not None:
    np.savez(args.all_params_file,
             camera_matrix=camera_matrix, distortion=distortion)

  print("Intrinsic calibration done.")

  return (camera_matrix, distortion)


def draw_test_pattern(frame):
  reverse_projector = reverse_projection.ReverseProjector(args.all_params_file)
  frame = reverse_projector.undistort(frame)

  num_columns = 100
  num_rows = 100
  for i in range(0, num_columns + 1):
    for j in range(0, num_rows + 1):
      u = 50 + ((frame.shape[1] - 100) // (num_columns - 1)) * i
      v = 50 + ((frame.shape[0] - 100) // (num_rows - 1)) * j

      if (i % 10) == 0 and (j % 10) == 0:
        [x, y] = reverse_projector.find_x_y(u, v, 0)

        frame = cv2.putText(frame, f"({int(x)}, {int(y)})", (u - 50, v + 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        frame = cv2.circle(frame, (u, v), 3, (0, 0, 255), 1, cv2.LINE_AA)
      else:
        frame = cv2.circle(frame, (u, v), 1, (0, 0, 255), 1, cv2.LINE_AA)
  return frame


def extrinsic_calibration(camera_matrix, distortion):
  def make_rotation_matrix(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)

  def get_aruco_corners(point_1, point_2):
    # Note, these values were taken from the doc from the calibration sheet.
    # Given point 1 and point 2, where are the corners?
    default_corners = np.array([
        # Marker 0:
        [[0, 0, 0],
         [100, 0, 0],
         [100, -100, 0],
         [0, -100, 0]],

        # Marker 1:
        [[0, -142, 0],
         [100, -142, 0],
         [100, -242, 0],
         [0, -242, 0]],
    ], np.float32)

    known_point_1 = np.array([[200 - 15], [-(32 - 20)], [0]], np.float32)
    known_point_2 = np.array([[197.5 - 15], [-(255.5 - 20)], [0]], np.float32)

    z_rotation = (math.atan2(point_2[1] - point_1[1], point_2[0] - point_1[0]) -
                  math.atan2(known_point_2[1][0] - known_point_1[1][0], known_point_2[0][0] - known_point_1[0][0]))
    rotation_matrix = make_rotation_matrix(z_rotation)
    translation = np.array([[point_1[0]], [point_1[1]], [0]]) - \
        rotation_matrix @ known_point_1
    return np.array([
        [(translation + rotation_matrix @ pt.reshape((3, 1))).reshape((3,))
         for pt in rect]
        for rect in default_corners], np.float32)

  def find_camera_transform_aruco(camera_matrix, distortion, image,
                                  point_1, point_2):
    image = cv2.undistort(image, camera_matrix, distortion)

    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_50)
    arucoParams = aruco.DetectorParameters_create()
    (corners, ids, _) = aruco.detectMarkers(
        image, arucoDict, parameters=arucoParams)

    board_corners = get_aruco_corners(point_1, point_2)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
    board = aruco.Board_create(
        board_corners, aruco_dict, np.array([0, 1]))

    (_, rvec, tvec) = aruco.estimatePoseBoard(
        corners, ids, board, camera_matrix, distortion, None, None)

    if args.debug:
      print("image corners:")
      print(corners)
      print("world corners:")
      print(board_corners)
      print("Rotation: ")
      print(rvec)
      print("Translation: ")
      print(tvec)

      print("Checking")

      reverse_projector = reverse_projection.ReverseProjector(
          args.all_params_file)
      for rect_ar in corners:
        for (u, v) in rect_ar[0]:
          print(reverse_projector.find_x_y(u, v, 0))

    return (rvec, tvec)

  if args.camera_position_photo is not None:
    if not os.path.exists(args.camera_position_photo):
      print("camera_position_photo does not exist.")
      exit(0)

    aruco_image = cv2.imread(args.camera_position_photo)
  else:
    print("Place the aruco calibraiton board flat on the plane you want to later measure.")
    image_ok = False
    while not image_ok:
      countdown(5)
      global show_camera
      show_camera = False
      aruco_image = new_frame
      cv2.imwrite("a.png", aruco_image)
      is_good = input("Does the image look good? [Y/n]: ")
      if is_good.lower() == 'y' or is_good.lower() == '':
        image_ok = True
      else:
        show_camera = True

    def read_param(name, default):
      val = input(f"{name} [{default}]:").strip()
      return default if val == "" else float(val)
    args.x1 = read_param("x1", args.x1)
    args.y1 = read_param("y1", args.y1)
    args.x2 = read_param("x2", args.x2)
    args.y2 = read_param("y2", args.y2)

  (camera_rotation, camera_translation) = find_camera_transform_aruco(
      camera_matrix, distortion, aruco_image, (args.x1, args.y1), (args.x2, args.y2))

  np.savez(args.all_params_file, camera_matrix=camera_matrix,
           distortion=distortion, camera_rotation=camera_rotation, camera_translation=camera_translation)

  show_camera = True
  global preview_display_image
  preview_display_image = draw_test_pattern(aruco_image)
  input("Calibration comlete, parameters written to file. Press enter to exit.")


def test_calibration():
  global show_camera, preview_display_image
  show_camera = True
  countdown(2)
  frame = new_frame

  preview_display_image = draw_test_pattern(frame)

  reverse_projector = reverse_projection.ReverseProjector(args.all_params_file)

  while True:
    x_str = input("X: ")
    if x_str == "":
      break
    x = float(x_str)
    y = float(input("Y: "))
    print(reverse_projector.find_x_y(x, y, 0))

  # input("Press enter to exit.")


if args.test_only:
  test_calibration()
else:
  if not args.skip_intrinsic_calibration:
    (camera_matrix, distortion) = calibrate_intrinsic()
  elif not args.skip_extrinsic_calibration:
    if args.intrinsic_params_file is not None and os.path.exists(args.intrinsic_params_file):
      npzfile = np.load(args.intrinsic_params_file)
      (camera_matrix, distortion) = (
          npzfile['camera_matrix'], npzfile['distortion'])
    elif args.all_params_file is not None and os.path.exists(args.all_params_file):
      npzfile = np.load(args.all_params_file)
      (camera_matrix, distortion) = (
          npzfile['camera_matrix'], npzfile['distortion'])
    else:
      print("intrinsic_params_file or all_params_file must exist if you're skipping intrinsic calibration.")
      exit(0)

  if not args.skip_extrinsic_calibration:
    extrinsic_calibration(camera_matrix, distortion)

stop_camera = True
camera_thread.join()
cv2.destroyAllWindows()
