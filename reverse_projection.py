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

  def find_x_y_undistorted(self, u, v, z):
    """Finds world X,Y coordinates given world Z and picture u,v coordinates of the undistorted image.

    Arguments:
    u -- x coordinate of the pixel on the un-distorted image.
    v -- y coordinate of hte pixel on the un-distorted image.
    z -- world z coordinate. +z is closer to the camera

    Returns:
    (x, y) in world coordinates.
    """
    uv = np.array([[u], [v], [1]])
    x_cam = self.inv_camera_matrix @ uv

    lhs = self.inv_rotation_matrix @ x_cam
    tmp = self.inv_rotation_matrix @ self.camera_translation
    s = (z + tmp[2][0]) / lhs[2][0]

    x_world = self.inv_rotation_matrix @ ((x_cam * s) -
                                          self.camera_translation)
    return [x_world[0][0], x_world[1][0]]

  def find_x_y_distorted(self, u, v, z):
    """Finds world X,Y coordinates given world Z and picture u,v coordinates of the original image.

    Arguments:
    u -- x coordinate of the pixel on the distorted image.
    v -- y coordinate of hte pixel on the distorted image.
    z -- world z coordinate. +z is closer to the camera

    Returns:
    (x, y) in world coordinates.
    """
    undistorted_normalized_point = cv2.undistortPoints(
        np.array([[u, v]], dtype=np.float32), self.camera_matrix, self.distortion)[0]

    # Convert [[x, y]] into [[x], [y], [1]]:
    undistorted_normalized_point = np.concatenate(
        [undistorted_normalized_point.reshape((2, 1)), np.array([[1]])])

    undistorted_point = self.camera_matrix @ undistorted_normalized_point

    return self.find_x_y_undistorted(undistorted_point[0][0], undistorted_point[1][0], z)
