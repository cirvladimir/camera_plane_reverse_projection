import math
import cv2
import numpy as np


class ReverseProjector():

  def __init__(self, camera_matrix, distortion, camera_rotation, camera_translation):
    self.camera_matrix = camera_matrix
    self.distortion = distortion
    self.camera_rotation = camera_rotation
    self.camera_translation = camera_translation

    self.rotation_matrix = cv2.Rodrigues(self.camera_rotation)[0]
    self.inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)
    self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)

  @classmethod
  def load(cls, calibration_file):
    npzfile = np.load(calibration_file)
    return cls(npzfile['camera_matrix'],
               npzfile['distortion'],
               npzfile['camera_rotation'],
               npzfile['camera_translation'])

  def undistort(self, image):
    return cv2.undistort(image, self.camera_matrix, self.distortion)

  def find_x_y_no_distortion(self, u, v, z):
    """Efficient version of find_x_y if the distortion of the camera is zero. Will give
    the wrong result if the distortion is not zero.
    """
    uv = np.array([[u], [v], [1]])
    x_cam = self.inv_camera_matrix @ uv

    lhs = self.inv_rotation_matrix @ x_cam
    tmp = self.inv_rotation_matrix @ self.camera_translation
    s = (z + tmp[2][0]) / lhs[2][0]

    x_world = self.inv_rotation_matrix @ ((x_cam * s) -
                                          self.camera_translation)
    return [x_world[0][0], x_world[1][0]]

  def find_x_y(self, u, v, z):
    """Finds world X,Y coordinates given world Z and picture u,v coordinates.

    Arguments:
    u -- x coordinate of the pixel on the un-distorted image.
    v -- y coordinate of hte pixel on the un-distorted image.
    z -- world z coordinate. +z is closer to the camera

    Returns:
    (x, y) in world coordinates.
    """
    initial_guess = np.array(self.find_x_y_no_distortion(u, v, z))

    desired = np.array([u, v])

    def get_actual(guess):
      imgage_points, _ = cv2.projectPoints(np.array([[[guess[0]], [guess[1]], [z]]]), self.camera_rotation, self.camera_translation, self.camera_matrix,
                                           self.distortion)
      return imgage_points[0][0]

    def get_error(guess):
      actual = get_actual(guess)
      return np.linalg.norm(actual - desired)

    best_guess = initial_guess
    best_error = get_error(initial_guess)
    travel_rad = 2
    for i in range(20):
      NUM_ANGLES = 8
      guess_improved = False
      for angle_i in range(NUM_ANGLES):
        angle = math.pi * 2 / NUM_ANGLES * angle_i
        new_guess = best_guess + \
            np.array([math.cos(angle), math.sin(angle)]) * travel_rad
        new_error = get_error(new_guess)
        if new_error < best_error:
          guess_improved = True
          best_error = new_error
          best_guess = new_guess
      if not guess_improved:
        travel_rad /= 2

    return best_guess
