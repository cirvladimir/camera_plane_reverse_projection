import cv2

from reverse_projection import ReverseProjector
from camera import Camera


def draw_test_pattern(frame, args):
  reverse_projector = ReverseProjector.load(args.extrinsic_params_file)
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


def test_calibration(args, camera: Camera):
  if args.image is not None:
    frame = cv2.imread(args.image)
  else:
    frame = camera.countdown_capture(delay_seconds=2)
    camera.stop()

  preview_display_image = draw_test_pattern(frame, args)
  cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
  cv2.imshow("preview", preview_display_image)
  cv2.waitKey(0)

  reverse_projector = ReverseProjector.load(args.extrinsic_params_file)

  while True:
    x_str = input("X: ")
    if x_str == "":
      break
    x = float(x_str)
    y = float(input("Y: "))
    print(reverse_projector.find_x_y_distorted(x, y, 0))
  cv2.destroyAllWindows()
