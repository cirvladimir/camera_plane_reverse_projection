# This utility heps capture a bunch of camera calibration files.

import argparse
import numpy as np
import cv2
import os
from calibration_viewer import test_calibration
from camera import Camera
from extrinsic_calibration import extrinsic_calibration
from intrinsic_calibration import intrinsic_calibration

parser = argparse.ArgumentParser(
    description='Utility for calibrating the camera.',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

subparsers = parser.add_subparsers(dest='command')

intrinsic_parser = subparsers.add_parser("intrinsic")
extrinsic_parser = subparsers.add_parser("extrinsic")
verification_parser = subparsers.add_parser("verification")
capture_parser = subparsers.add_parser("capture")
resize_calibration_parser = subparsers.add_parser("resize_calibration")


parser.add_argument(
    "--camera",
    default=0,
    type=int,
    help="Index/device name of camera. Usually 0 for default opencv setups. Corresponds to cv2.VideoCapture(index)."
)

parser.add_argument(
    "--resolution",
    default=1600,
    type=int,
    help="Width of the resolution of the camera."
)

parser.add_argument("--debug",
                    action='store_true',
                    help="Print extra debug info.")

intrinsic_parser.add_argument(
    "--grid_width",
    default=8,
    type=int,
    help="Number of squares along the width of the calibration grid.")

intrinsic_parser.add_argument(
    "--grid_height",
    default=10,
    type=int,
    help="Number of squares along the height of the calibration grid.")

intrinsic_parser.add_argument(
    "--sharpness_threshold",
    default=50,
    type=int,
    help="Discard images which are less sharp than this value. Higher number discards more images, 0 turns feature off."
)

intrinsic_parser.add_argument("--num_images",
                              default=20,
                              type=int,
                              help="Number of camera distortion calibration photos.")

intrinsic_parser.add_argument("out_file",
                              help="Where to write intrinsic camera parameters. Recommend .npz extension.")


extrinsic_parser.add_argument("--image",
                              help="Specify this to use an image instead of capturing from the camera. Leave out to use camera.")

extrinsic_parser.add_argument("--x1",
                              default=185.0,
                              type=float,
                              help="X-coordinate of point 1 on aruco calibration image. Only needed if you're using camera_position_photo.")

extrinsic_parser.add_argument("--y1",
                              default=-12.0,
                              type=float,
                              help="Y-coordinate of point 1 on aruco calibration image. Only needed if you're using camera_position_photo.")

extrinsic_parser.add_argument("--x2",
                              default=182.5,
                              type=float,
                              help="X-coordinate of point 2 on aruco calibration image. Only needed if you're using camera_position_photo.")

extrinsic_parser.add_argument("--y2",
                              default=-235.5,
                              type=float,
                              help="Y-coordinate of point 2 on aruco calibration image. Only needed if you're using camera_position_photo.")

extrinsic_parser.add_argument("intrinsic_params_file",
                              help="File in which the intrinsic parameters are stored from intrinci calibration.")

extrinsic_parser.add_argument("out_file",
                              help="Where to write extrinsic + intrinsic camera parameters. Intrinsic parameters copied to this file for convenience. Recommend .npz extension.")

verification_parser.add_argument("--image",
                                 help="Specify this to use an image instead of capturing from the camera. Leave out to use camera.")

verification_parser.add_argument(
    "extrinsic_params_file", help="Extrinsic calibration file.")

capture_parser.add_argument("out_image",
                            help="Take a photo and write it to this path.")

capture_parser.add_argument(
    "--params_file", help="Optional. If specified, intrinsic camera parameters will be read from this file and the image will be undistorted.")

resize_calibration_parser.add_argument(
    "--old_params", help="Old intrinsic parameters file.")

resize_calibration_parser.add_argument(
    "--new_params", help="New parameters will be written to this file.")

resize_calibration_parser.add_argument(
    "--old_resolution", type=int, help="Width of the old resolution.")

resize_calibration_parser.add_argument(
    "--new_resolution", type=int, help="Width of the new resolution.")

parser.epilog = f"commands usage:\n{intrinsic_parser.format_usage().replace('usage: ', '')}{extrinsic_parser.format_usage().replace('usage: ', '')}{verification_parser.format_usage().replace('usage: ', '')}{capture_parser.format_usage().replace('usage: ', '')}{resize_calibration_parser.format_usage().replace('usage: ', '')}"


def main():
  args = parser.parse_args()

  camera = None
  need_camera = False

  if args.command == 'intrinsic':
    need_camera = True
  elif args.command == 'extrinsic':
    need_camera = args.image is None
  elif args.command == 'verification':
    need_camera = args.image is None
  elif args.command == 'capture':
    need_camera = True
  elif args.command == 'resize_calibration':
    need_camera = False
  else:
    parser.print_help()
    return

  if need_camera:
    camera = Camera(args)
    camera.start()

  if args.command == 'intrinsic':
    intrinsic_calibration(args, camera)
  elif args.command == 'extrinsic':
    if not os.path.exists(args.intrinsic_params_file):
      print("Error: The intrinsic parameter file does not exist.")
      return
    extrinsic_calibration(args, camera)
  elif args.command == 'verification':
    test_calibration(args, camera)
  elif args.command == 'capture':
    image = camera.countdown_capture(1)
    if args.params_file is not None:
      npzfile = np.load(args.params_file)
      (camera_matrix, distortion) = (
          npzfile['camera_matrix'], npzfile['distortion'])
      image = cv2.undistort(image, camera_matrix, distortion)
    cv2.imwrite(args.out_image, image)
  elif args.command == 'resize_calibration':
    npzfile = np.load(args.old_params)
    (camera_matrix, distortion) = (
        npzfile['camera_matrix'], npzfile['distortion'])
    camera_matrix = camera_matrix * args.new_resolution / args.old_resolution
    camera_matrix[2, 2] = 1.0
    np.savez(args.new_params,
             camera_matrix=camera_matrix, distortion=distortion)

  if camera is not None:
    camera.release()


if __name__ == "__main__":
  main()
