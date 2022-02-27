import cv2
import numpy as np


class ReverseProjector():

  def __init__(self, calibration_file):
    npzfile = np.load(calibration_file)

    self.camera_matrix = npzfile['camera_matrix']
    self.distortion = npzfile['distortion']
    self.camera_rotation = npzfile['camera_rotation']
    self.camera_translation = npzfile['camera_translation']

    self.rotation_matrix = cv2.Rodrigues(self.camera_rotation)[0]
    self.inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)
    self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)

  def undistort(self, image):
    return cv2.undistort(image, self.camera_matrix, self.distortion)

  def find_x_y(self, u, v, z):
    """Finds world X,Y coordinates given world Z and picture u,v coordinates.

    Arguments:
    u -- x coordinate of the pixel on the un-distorted image.
    v -- y coordinate of hte pixel on the un-distorted image.
    z -- world z coordinate. +z is closer to the camera

    Returns:
    (x, y) in world coordinates.
    """

    uv = np.array([[u], [v], [1]])
    lsm = np.dot(np.dot(self.inv_rotation_matrix, self.inv_camera_matrix), uv)
    rsm = np.dot(self.inv_rotation_matrix, self.camera_translation)

    s = (z + rsm[2, 0]) / lsm[2, 0]
    p = np.dot(self.inv_rotation_matrix, s *
               np.dot(self.inv_camera_matrix, uv) - self.camera_translation)

    # TODO: Fix this hack! Should be 0, 1 bellow:
    return np.array([p[1], p[0]])
