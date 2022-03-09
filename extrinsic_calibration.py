import math
import os
import cv2
import numpy as np
from cv2 import aruco
from camera import Camera

from reverse_projection import ReverseProjector


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
                                point_1, point_2, debug=False):
  arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_50)
  arucoParams = aruco.DetectorParameters_create()
  (corners, ids, _) = aruco.detectMarkers(
      image, arucoDict, parameters=arucoParams, cameraMatrix=camera_matrix, distCoeff=distortion)

  board_corners = get_aruco_corners(point_1, point_2)

  aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
  board = aruco.Board_create(
      board_corners, aruco_dict, np.array([0, 1]))

  (_, rvec, tvec) = aruco.estimatePoseBoard(
      corners, ids, board, camera_matrix, distortion, None, None)

  reprojected_points, _ = cv2.projectPoints(
      np.reshape(board_corners, (len(board_corners) * 4, 3, 1)), rvec, tvec, camera_matrix, distortion)

  reprojected_points = np.reshape(
      reprojected_points, (len(board_corners) * 4, 2))

  reshaped_corners = np.reshape(
      [corners[ids[0][0]], corners[ids[1][0]]], (len(board_corners) * 4, 2))

  difs = np.linalg.norm(reprojected_points - reshaped_corners, axis=1)
  pixel_size = np.linalg.norm(
      board_corners[0][0] - board_corners[0][1]) / np.linalg.norm(reshaped_corners[1] - reshaped_corners[0])

  print(
      f"Average error: {np.mean(difs) * pixel_size}, Max error: {np.max(difs) * pixel_size}")

  if debug:
    print("image corners:")
    print(corners)
    print("world corners:")
    print(board_corners)
    print("Rotation: ")
    print(rvec)
    print("Translation: ")
    print(tvec)
    print("Checking")

    reverse_projector = ReverseProjector(camera_matrix, distortion,
                                         rvec, tvec)
    for rect_ar in corners:
      for (u, v) in rect_ar[0]:
        print(reverse_projector.find_x_y_distorted(u, v, 0))

  return (rvec, tvec)


def extrinsic_calibration(args, camera: Camera):
  npzfile = np.load(args.intrinsic_params_file)
  (camera_matrix, distortion) = (
      npzfile['camera_matrix'], npzfile['distortion'])

  if args.image is not None:
    if not os.path.exists(args.image):
      print("Specified image does not exist.")
      return

    aruco_image = cv2.imread(args.image)
  else:
    print("Place the aruco calibraiton board flat on the plane you want to later measure.")
    aruco_image = camera.countdown_capture(delay_seconds=5)

    def read_param(name, default):
      val = input(f"{name} [{default}]:").strip()
      return default if val == "" else float(val)
    args.x1 = read_param("x1", args.x1)
    args.y1 = read_param("y1", args.y1)
    args.x2 = read_param("x2", args.x2)
    args.y2 = read_param("y2", args.y2)

  (camera_rotation, camera_translation) = find_camera_transform_aruco(
      camera_matrix, distortion, aruco_image, (args.x1, args.y1), (args.x2, args.y2), debug=args.debug)

  np.savez(args.out_file, camera_matrix=camera_matrix,
           distortion=distortion, camera_rotation=camera_rotation, camera_translation=camera_translation)
